# Vendored astrollm (ONNX inference, torch-free)

OriginStack's `--astrollm` advisory scoring shells out to `infer_onnx.py` here.
The exported `model.onnx` runs under `onnxruntime` -- no torch, no astropy in
this vendored copy or its venv.

## One-time setup

```powershell
cd vendor/astrollm
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt   # onnxruntime + opencv + numpy (~80 MB)
```

## Use

Nothing to pass -- when `vendor/astrollm/` exists, `--astrollm` auto-resolves
`--astrollm-dir` to it (`--astrollm-python` = `vendor/astrollm/.venv/...`,
`--astrollm-script` = `infer_onnx.py`, `--astrollm-checkpoint` =
`checkpoints/model.onnx`). Point `--astrollm-dir` elsewhere to override.

```bash
python originstack.py -d session/ -o out.fits --astrollm
```

`infer_onnx.py` takes TIFF/PNG/JPG, not FITS. OriginStack debayers each raw
light frame to a temp image first (`src/astrollm.py::_render_light_for_onnx`)
-- a small score-drift vs astrollm's own cv2 debayer, acceptable for an
advisory-only signal. The final stacked master is already a rendered image and
is passed straight through.

## This checkpoint

`model.onnx` is the epoch-15 SSL-pretrained + fine-tuned model. Its
meaningful heads (ONNX metadata `tasks`): `reject`, `quality`, `category`,
`exposure`, `sky_brightness`, `stray_light_gradient`. `category` is a 4-class
head -- `galaxy`, `nebula`, `star_cluster`, `comet` -- and its `comet`
prediction is shape-gated (`data/shape_features.py`, comet precision
0.33 -> 0.82). `exposure` is a 7-class classifier (5/10/15/20/25/30/40 s).

See `VENDORED_FROM.txt` for the exact upstream commit and re-sync steps.
