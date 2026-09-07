# Vendored originvision — upstream snapshot (provenance only)

**This directory is not on OriginStack's runtime path.** originvision scoring
(`--originvision`) runs in-process from [../../src/originvision_infer.py](../../src/originvision_infer.py)
(a numpy/scipy port of the files here, no OpenCV) against the model bundled at
[../../src/data/originvision.onnx](../../src/data/originvision.onnx). `onnxruntime` is
the only extra dependency, and it's optional.

What's kept here is the upstream source the port and the bundled `.onnx` were
copied from, so a future re-sync can diff against it:

| file | upstream role |
|------|---------------|
| `infer_onnx.py` | torch-free ONNX entry point — the reference for `src/originvision_infer.py` |
| `data/imageops.py` | percentile stretch + resize/centre-crop + background correction |
| `data/shape_features.py` | comet shape-gate features |
| `checkpoints/model.onnx` | the exported model (byte-identical to `src/data/originvision.onnx`) |

## Current model (v4)

`model_v4_nossl_best.pt` → ONNX, epoch 15, a **from-scratch** (no-SSL) 7-task
run. The graph emits **8** outputs (`head_order` always builds `trailing` +
`background_grid`) but ONNX metadata `tasks` lists **7** — `trailing` is
untrained on this checkpoint. `src/originvision_infer.py` gates every head on
`tasks`, so `trailing` (noise) and `background_grid` (trained, but unused —
OriginStack has DBE) are not surfaced. See `VENDORED_FROM.txt`.

## Re-syncing to a newer originvision

1. Copy the four source files above from the new upstream commit into this dir.
2. `cp checkpoints/model.onnx ../../src/data/originvision.onnx`.
3. Port any `infer_onnx.py` / `imageops.py` / `shape_features.py` logic changes
   into `src/originvision_infer.py` (numpy/scipy, no cv2) and re-run
   `tests/test_originvision.py` — the `TestRealInference` class parity-checks the
   port against a real forward pass.
4. Update `VENDORED_FROM.txt` (commit hash + notes).

See `VENDORED_FROM.txt` for the exact upstream commit currently vendored.
