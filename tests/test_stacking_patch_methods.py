"""The patch-weighted combine only reproduces mean and the rejection methods
it builds a mask for. It used to run for every --stack-method whenever
quality maps existed (--auto turns patch weighting on at 15+ frames), so a
requested median/linear_fit/ivw/wavelet silently became an unrejected
weighted mean: a single cosmic ray survived at ~1/N of its amplitude."""
import numpy as np
import pytest

from src import stacking as st
from src.cli import parse_args
from src.models import FrameInfo, ProcessingStats


def _stack(method, capsys):
    n, h, w, c = 12, 48, 48, 3
    args = parse_args(['-d', 'x', '-o', 'y.fits', '--stack-method', method, '--no-auto'])
    rng = np.random.default_rng(0)
    mem = (100 + rng.normal(0, 1, (n, h, w, c))).astype(np.float32)
    mem[3, 24, 24, :] = 10000.0          # one cosmic ray in one frame
    final = [FrameInfo(path=f'f{i}.fits', type='light', header={},
                       metrics={'score': 1.0, 'noise': 1.0, 'fwhm': 3.0}) for i in range(n)]
    qmaps = [np.ones((4, 4), np.float32) for _ in range(n)]
    _, fits_stacked, top, _, left, _ = st.run_stacking_phase(
        final, list(range(n)), mem, [(0.0, 0.0)] * n, [None] * n, h, w, c, args,
        ProcessingStats(), quality_maps=qmaps)
    return fits_stacked[24 - top, 24 - left], capsys.readouterr().out


@pytest.mark.parametrize('method', ['median', 'linear_fit'])
def test_non_patch_methods_keep_their_own_rejection(method, capsys):
    px, out = _stack(method, capsys)
    assert np.all(np.abs(px - 100.0) < 5.0), px     # was ~925
    assert 'patch weighting is not available' in out


@pytest.mark.parametrize('method', ['sigma_clip', 'percentile'])
def test_rejection_methods_still_use_the_patch_path(method, capsys):
    px, out = _stack(method, capsys)
    assert np.all(np.abs(px - 100.0) < 5.0), px
    assert 'Patch-weighted mean combine' in out
