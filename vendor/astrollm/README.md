# Vendored astrollm — upstream snapshot (provenance only)

**This directory is not on OriginStack's runtime path.** astrollm scoring
(`--astrollm`) runs in-process from [../../src/astrollm_infer.py](../../src/astrollm_infer.py)
(a numpy/scipy port of the files here, no OpenCV) against the model bundled at
[../../src/data/astrollm.onnx](../../src/data/astrollm.onnx). `onnxruntime` is
the only extra dependency, and it's optional.

What's kept here is the upstream source the port and the bundled `.onnx` were
copied from, so a future re-sync can diff against it:

| file | upstream role |
|------|---------------|
| `infer_onnx.py` | torch-free ONNX entry point — the reference for `src/astrollm_infer.py` |
| `data/imageops.py` | percentile stretch + resize/centre-crop + background correction |
| `data/shape_features.py` | comet shape-gate features |
| `checkpoints/model.onnx` | the exported model (byte-identical to `src/data/astrollm.onnx`) |

## Re-syncing to a newer astrollm

1. Copy the four source files above from the new upstream commit into this dir.
2. `cp checkpoints/model.onnx ../../src/data/astrollm.onnx`.
3. Port any `infer_onnx.py` / `imageops.py` / `shape_features.py` logic changes
   into `src/astrollm_infer.py` (numpy/scipy, no cv2) and re-run
   `tests/test_astrollm.py` — the `TestRealInference` class parity-checks the
   port against a real forward pass.
4. Update `VENDORED_FROM.txt` (commit hash + notes).

See `VENDORED_FROM.txt` for the exact upstream commit currently vendored.
