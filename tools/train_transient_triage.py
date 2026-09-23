"""Train the ZOGY transient-triage model from synthetic data
(tools/gen_transient_triage_data.py) and export it to ONNX
(src/data/transient_triage.onnx, consumed by
astro_native.transient_triage_score / src/transient_triage.py).

``torch`` is a script-local optional dependency, not part of this project's
runtime dependencies -- model training happens outside the shipped package,
the same stance ``src/data/originvision.onnx`` itself was trained under (see
vendor/originvision/README.md).

The model is deliberately small (a handful of conv layers) given the tiny
31x31x3 input and a synthetic-only training set -- there is no reason to
reach for originvision's 256x256-real-photograph-classifier capacity here.

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
        "the shipped package -- see vendor/originvision/README.md for the "
        "same stance on originvision.onnx). Install it with: pip install torch"
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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default=None,
                        help='.npz from gen_transient_triage_data.py '
                             '(default: tools/../transient_triage_data.npz)')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val-frac', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', default=None,
                        help='Output ONNX path (default: src/data/transient_triage.onnx)')
    args = parser.parse_args()

    data_path = args.data or os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'transient_triage_data.npz'))
    out_path = args.out or os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'src', 'data', 'transient_triage.onnx'))

    npz = np.load(data_path)
    X, y, size = npz['X'], npz['y'], int(npz['size'])
    n = len(y)
    if n < 50:
        raise SystemExit(f"only {n} labelled stamps in {data_path} -- generate "
                         f"more with tools/gen_transient_triage_data.py first")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * args.val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TriageNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).float()

    def _batches(idx):
        order = idx.copy()
        rng.shuffle(order)
        for i in range(0, len(order), args.batch_size):
            b = order[i:i + args.batch_size]
            yield X_t[b].to(device), y_t[b].to(device)

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
            val_pred = (torch.sigmoid(model(X_t[val_idx].to(device))) > 0.5).float()
            val_acc = float((val_pred.cpu() == y_t[val_idx]).float().mean())
        n_train = max(1, len(train_idx))
        print(f'epoch {epoch + 1}/{args.epochs}  '
             f'train_loss={total_loss / n_train:.4f}  val_acc={val_acc:.3f}')

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
    # itself), but it mirrors originvision.onnx's own metadata_props and
    # matters for a future re-train/re-sync. Best-effort: onnx is not a
    # project dependency either.
    try:
        import onnx
        m = onnx.load(out_path)
        for k, v in {'stamp_size': str(size), 'channels': 'new,ref,diff',
                    'trained_on': 'synthetic'}.items():
            e = m.metadata_props.add()
            e.key, e.value = k, v
        onnx.save(m, out_path)
    except ImportError:
        print("(skipping ONNX metadata -- `pip install onnx` to include it)")

    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
