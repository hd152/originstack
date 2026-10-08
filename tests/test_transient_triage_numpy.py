"""numpy backend of ``--transient-triage`` (``src/transient_triage.py``).

The numpy forward pass reads the bundled ONNX model with its own protobuf
reader; these tests pin it against onnxruntime and the native kernel (each
skipped when absent) and check it refuses graphs it does not implement.
"""
import numpy as np
import pytest

import src.transient_triage as tt

MODEL = tt.bundled_model_path()


def _stamps(n=40, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3, 31, 31)).astype(np.float32)
    yy, xx = np.mgrid[:31, :31]
    blob = np.exp(-((yy - 15) ** 2 + (xx - 15) ** 2) / (2 * 1.8 ** 2)).astype(np.float32)
    amp = rng.uniform(0, 60, n).astype(np.float32)
    x[:, 0] += amp[:, None, None] * blob
    x[:, 2] += amp[:, None, None] * blob
    x[::5, 2, 15, 15] += 80.0  # cosmic-ray-like spikes on some
    return x


def test_numpy_matches_onnxruntime():
    ort = pytest.importorskip('onnxruntime')
    x = _stamps()
    sess = ort.InferenceSession(MODEL, providers=['CPUExecutionProvider'])
    logit = sess.run(None, {'stamps': x})[0].astype(np.float64)
    ref = 1.0 / (1.0 + np.exp(-logit))
    np.testing.assert_allclose(tt._score_numpy(x, MODEL), ref, atol=1e-5)


@pytest.mark.skipif(not tt._HAS_NATIVE_TRIAGE, reason="astro_native built without 'triage'")
def test_numpy_matches_native():
    x = _stamps()
    native = np.asarray(tt._native.transient_triage_score(x, MODEL, 31))
    np.testing.assert_allclose(tt._score_numpy(x, MODEL), native, atol=1e-5)


def test_batches_are_independent():
    """Chunking (64 per pass) must not change any candidate's score."""
    x = _stamps(n=150)
    full = tt._score_numpy(x, MODEL)
    np.testing.assert_allclose(tt._score_numpy(x[100:110], MODEL), full[100:110], rtol=0, atol=1e-12)


def test_rejects_a_different_architecture(tmp_path):
    onnx = pytest.importorskip('onnx')
    m = onnx.load(MODEL)
    relu = [n for n in m.graph.node if n.op_type == 'Relu'][0]
    relu.op_type = 'Sigmoid'
    bad = tmp_path / 'bad.onnx'
    onnx.save(m, str(bad))
    with pytest.raises(ValueError, match='unsupported triage graph'):
        tt.load_onnx_weights(str(bad))


def test_garbage_file_is_a_clean_error(tmp_path):
    bad = tmp_path / 'garbage.onnx'
    bad.write_bytes(b'\xff' * 64)
    with pytest.raises(ValueError):
        tt.load_onnx_weights(str(bad))
