"""--cfa-drizzle: combine the measured Bayer samples, not the interpolated ones."""
import argparse

import numpy as np
import pytest
from scipy import ndimage as ndi

from src.cfa_drizzle import apply_cfa_drizzle, cfa_drizzle_combine, cfa_lattice
from src.debayer import debayer
from src.registration import apply_transform


def test_lattice_covers_each_site_once():
    for pattern in ('RGGB', 'bggr', 'GRBG', 'GBRG'):
        lat = cfa_lattice(pattern, 8, 10)
        total = sum(len(iy) for iy, _ in lat)
        assert total == 80
        assert [len(lat[c][0]) for c in range(3)] == [20, 40, 20]
        seen = np.zeros((8, 10), int)
        for iy, ix in lat:
            seen[iy, ix] += 1
        assert (seen == 1).all()
    iy, ix = cfa_lattice('RGGB', 4, 4)[0]           # R only at even/even
    assert set(zip(iy.tolist(), ix.tolist())) == {(0, 0), (0, 2), (2, 0), (2, 2)}
    iy, ix = cfa_lattice('BGGR', 4, 4)[0]           # R only at odd/odd
    assert set(zip(iy.tolist(), ix.tolist())) == {(1, 1), (1, 3), (3, 1), (3, 3)}


def test_bad_pattern_rejected():
    with pytest.raises(ValueError):
        cfa_lattice('RGB', 4, 4)
    with pytest.raises(ValueError):
        cfa_lattice('RGXB', 4, 4)


def _sim(n_frames, seed=1, H=96, SS=4, pat='RGGB', noise=40.0):
    """Dithered noisy Bayer frames of one synthetic star field."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:H * SS, :H * SS] / SS
    hi = np.zeros((H * SS, H * SS, 3))
    for _ in range(40):
        cy, cx = rng.uniform(10, H - 10, 2)
        g = rng.uniform(1500, 9000) * np.exp(
            -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * rng.uniform(1.3, 2.0) ** 2))
        col = (1, .8, .6) if rng.random() < .5 else (.7, .85, 1)
        for c in range(3):
            hi[:, :, c] += g * col[c]
    hi += 1000
    native = lambda a: a.reshape(H, SS, H, SS, 3).mean((1, 3))
    truth = native(hi)
    frames, shifts = [], []
    for _ in range(n_frames):
        sy, sx = rng.integers(-12, 13, 2) / 4.0
        nat = native(ndi.shift(hi, (sy * SS, sx * SS, 0), order=1, mode='nearest'))
        m = np.zeros((H, H), np.float32)
        m[0::2, 0::2] = nat[0::2, 0::2, 0]
        m[0::2, 1::2] = nat[0::2, 1::2, 1]
        m[1::2, 0::2] = nat[1::2, 0::2, 1]
        m[1::2, 1::2] = nat[1::2, 1::2, 2]
        m += rng.normal(0, noise, m.shape).astype(np.float32)
        frames.append(debayer(m, pattern=pat, method='malvar'))
        shifts.append((-float(sy), -float(sx)))
    return truth, np.stack(frames), shifts


def test_beats_debayer_then_stack_on_star_cores():
    n = 100
    truth, mem, shifts = _sim(n)
    std = np.mean([apply_transform(f, shift=s) for f, s in zip(mem, shifts)],
                  axis=0).astype(np.float32)
    out, stats = cfa_drizzle_combine(mem, list(range(n)), shifts, [None] * n, std,
                                     'RGGB', 0, 0, scale=1.0, pixfrac=0.6)
    b = 12
    core = truth[:, :, 1] > 3000
    core[:b] = core[-b:] = False
    core[:, :b] = core[:, -b:] = False
    err = lambda x: float(np.sqrt(np.mean((x - truth)[core] ** 2)))
    assert err(out) < 0.85 * err(std)
    assert stats['frames'] == n
    assert all(f > 0.9 for f in stats['cfa_fraction'])


def test_rejects_a_hot_sample_without_touching_the_rest():
    n = 30
    truth, mem, shifts = _sim(n, seed=3)
    std = np.mean([apply_transform(f, shift=s) for f, s in zip(mem, shifts)],
                  axis=0).astype(np.float32)
    clean, _ = cfa_drizzle_combine(mem, list(range(n)), shifts, [None] * n, std,
                                   'RGGB', 0, 0, pixfrac=1.0)
    bad = mem.copy()
    bad[5, 48, 48, 0] = 60000.0            # R site: even/even -> a cosmic ray
    hit, stats = cfa_drizzle_combine(bad, list(range(n)), shifts, [None] * n, std,
                                     'RGGB', 0, 0, pixfrac=1.0)
    assert stats['rejected_frac'] > 0
    assert float(np.abs(hit - clean)[44:53, 44:53, 0].max()) < 50.0


def test_skips_cleanly_when_it_cannot_model_the_input(capsys):
    stacked = np.ones((8, 8, 3), np.float32)
    mem = np.ones((2, 8, 8, 3), np.float32)
    args = argparse.Namespace(_session_bayer=None, debayer_method='malvar',
                              drizzle_scale=1.0, drizzle_pixfrac=1.0)
    assert apply_cfa_drizzle(stacked, mem, [0, 1], [(0, 0)] * 2, [None] * 2, 0, 0, args) is stacked
    args._session_bayer = 'RGGB'
    args.debayer_method = 'menon2007'
    assert apply_cfa_drizzle(stacked, mem, [0, 1], [(0, 0)] * 2, [None] * 2, 0, 0, args) is stacked
    args.debayer_method = 'malvar'
    assert apply_cfa_drizzle(stacked, mem, [0, 1], [(0, 0)] * 2, [None] * 2, 0, 0, args,
                             displacement_fields=[np.zeros((8, 8, 2))] * 2) is stacked
    out = capsys.readouterr().out
    assert 'no Bayer pattern' in out and 'malvar' in out and 'elastic' in out
