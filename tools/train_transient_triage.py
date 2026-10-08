"""Train the ZOGY transient-triage model from synthetic data
(tools/gen_transient_triage_data.py) and export it to ONNX
(src/data/transient_triage.onnx, consumed by
astro_native.transient_triage_score / src/transient_triage.py).

``torch`` is a script-local optional dependency, not part of this project's
runtime dependencies -- model training happens outside the shipped package.

The model is deliberately small (a handful of conv layers) given the tiny
31x31x3 input and a synthetic-only training set.

Usage:
    pip install torch onnx
    python tools/gen_transient_triage_data.py --n-pairs 4000
    python tools/train_transient_triage.py
"""
from __future__ import annotations

import argparse
import os

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover - environment-dependent
    raise SystemExit(
        "tools/train_transient_triage.py needs torch, which is not part of "
        "this project's runtime dependencies (model training happens outside "
        "the shipped package). Install it with: pip install torch"
    ) from exc


class TriageNet(nn.Module):
    """Small conv net -> one logit. The native kernel applies sigmoid itself,
    so this exports a raw logit, matching astro_native's `compute()`."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.fc(x).squeeze(-1)


def _auc(p: np.ndarray, y: np.ndarray) -> float:
    """Rank-based ROC AUC (Mann-Whitney U); NaN if one class is missing."""
    pos = y > 0.5
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float('nan')
    from scipy.stats import rankdata
    r = rankdata(p)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', nargs='+', default=None,
                        help='.npz files from gen_transient_triage_data.py (synthetic) and/or '
                             'gen_transient_triage_real.py (real night pairs; carry a '
                             '"group" per target pair) (default: tools/../transient_triage_data.npz)')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val-frac', type=float, default=0.15)
    parser.add_argument('--holdout-group', type=int, default=None,
                        help='Validate on this real target pair (group index, counted '
                             'across the real files in order) and train on everything else')
    parser.add_argument('--no-export', action='store_true',
                        help='Train and report only (cross-validation folds)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default=None,
                        help='Output ONNX path (default: src/data/transient_triage.onnx)')
    args = parser.parse_args()

    data_paths = args.data or [os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'transient_triage_data.npz'))]
    out_path = args.out or os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'src', 'data', 'transient_triage.onnx'))

    Xs, ys, gs = [], [], []
    size, n_groups, has_real = None, 0, False
    for path in data_paths:
        npz = np.load(path)
        size = int(npz['size']) if size is None else size
        if int(npz['size']) != size:
            raise SystemExit(f'{path}: stamp size {int(npz["size"])} != {size}')
        Xs.append(npz['X'])
        ys.append(npz['y'])
        if 'group' in npz:      # real pairs: one group per target pair
            g = npz['group'].astype(np.int64)
            gs.append(g + n_groups)
            n_groups += int(g.max()) + 1
            has_real = True
        else:
            gs.append(np.full(len(npz['y']), -1, np.int64))
    X, y, grp = np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)
    n = len(y)
    if n < 50:
        raise SystemExit(f"only {n} labelled stamps -- generate more with "
                         f"tools/gen_transient_triage_data.py first")

    rng = np.random.default_rng(args.seed)
    if args.holdout_group is not None:
        val_idx = np.flatnonzero(grp == args.holdout_group)
        train_idx = np.flatnonzero(grp != args.holdout_group)
    else:
        perm = rng.permutation(n)
        n_val = max(1, int(n * args.val_frac))
        val_idx, train_idx = perm[:n_val], perm[n_val:]
    n_pos = float(y[train_idx].sum())
    pos_weight = (len(train_idx) - n_pos) / max(n_pos, 1.0)
    print(f'{len(train_idx)} train / {len(val_idx)} val stamps, '
          f'train positives {int(n_pos)} (pos_weight {pos_weight:.1f})')

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TriageNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).float()

    def _batches(idx):
        order = idx.copy()
        rng.shuffle(order)
        for i in range(0, len(order), args.batch_size):
            b = order[i:i + args.batch_size]
            xb = X_t[b]
            # The stamp's physics has no preferred orientation: random flips and
            # quarter turns are free extra data.
            k = int(rng.integers(4))
            if k:
                xb = torch.rot90(xb, k, dims=(2, 3))
            if rng.integers(2):
                xb = torch.flip(xb, dims=(3,))
            yield xb.to(device), y_t[b].to(device)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in _batches(train_idx):
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.detach().item() * len(yb)
        model.eval()
        with torch.no_grad():
            val_p = torch.sigmoid(model(X_t[val_idx].to(device))).cpu().numpy()
        val_acc = float(((val_p > 0.5) == (y[val_idx] > 0.5)).mean())
        n_train = max(1, len(train_idx))
        print(f'epoch {epoch + 1}/{args.epochs}  '
              f'train_loss={total_loss / n_train:.4f}  val_acc={val_acc:.3f}  '
              f'val_auc={_auc(val_p, y[val_idx]):.3f}')

    if args.no_export:
        return

    model.eval()
    model.to('cpu')  # export from CPU regardless of training device
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    dummy = torch.zeros(1, 3, size, size)
    torch.onnx.export(
        model, dummy, out_path,
        input_names=['stamps'], output_names=['logit'],
        dynamic_axes={'stamps': {0: 'batch'}, 'logit': {0: 'batch'}},
        opset_version=13,
        # The newer dynamo-based exporter (torch's default since 2.x) needs
        # `onnxscript`, an extra dependency beyond torch itself; the legacy
        # TorchScript-tracing exporter doesn't and is plenty for this model.
        dynamo=False,
    )

    # Lightweight provenance metadata -- astro_native's kernel doesn't read
    # it (it takes `size` as an explicit call argument and applies sigmoid
    # itself), but it records how the model was made, for a future re-train. Best-effort: onnx is not a
    # project dependency either.
    try:
        import onnx
        m = onnx.load(out_path)
        for k, v in {'stamp_size': str(size), 'channels': 'new,ref,diff',
                    'trained_on': 'synthetic+real' if has_real else 'synthetic'}.items():
            e = m.metadata_props.add()
            e.key, e.value = k, v
        onnx.save(m, out_path)
    except ImportError:
        print("(skipping ONNX metadata -- `pip install onnx` to include it)")

    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
