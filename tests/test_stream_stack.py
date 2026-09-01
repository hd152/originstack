"""Tests for src/stream_stack.py — the --stream two-pass genuine streaming stack."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from src.pipeline import _MemmapManager
from src.stream_stack import (
    _HAS_SCIPY,
    FrameRecord,
    fold,
    run_stream_stack,
    select_reference,
    survey,
)

pytestmark = pytest.mark.skipif(not _HAS_SCIPY, reason="scipy not installed")


def _write_star_frame(path, H=160, W=160, shift=(0, 0), n_stars=45, seed=0,
                      exptime=30.0, hit=None):
    """Write a (3,H,W) RGB FITS star field (+ broad nebula), optionally shifted.
    ``hit``: optional (y, x, amplitude) to inject a single bright outlier pixel
    (simulates a satellite trail / cosmic ray hit for rejection tests)."""
    from astropy.io import fits
    rng = np.random.default_rng(seed)
    img = np.full((H, W), 1000.0, np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    img += (300.0 * np.exp(-(((xx - W / 2) / (W * 0.3)) ** 2
                            + ((yy - H / 2) / (H * 0.3)) ** 2))).astype(np.float32)
    gy, gx = np.mgrid[-4:5, -4:5]
    g = np.exp(-(gx * gx + gy * gy) / (2 * 1.5 ** 2))
    star_rng = np.random.default_rng(999)
    for _ in range(n_stars):
        y0 = star_rng.integers(20, H - 20) + shift[0]
        x0 = star_rng.integers(20, W - 20) + shift[1]
        if 4 <= y0 < H - 4 and 4 <= x0 < W - 4:
            img[y0 - 4:y0 + 5, x0 - 4:x0 + 5] += (5000 * g).astype(np.float32)
    img += rng.standard_normal((H, W)).astype(np.float32) * 5.0
    if hit is not None:
        hy, hx, amp = hit
        img[hy, hx] += amp
    cube = np.stack([img, img, img], axis=2).astype(np.float32)  # (H,W,3) HWC
    hdu = fits.PrimaryHDU(data=cube)
    hdu.header['EXPTIME'] = exptime
    hdu.writeto(path, overwrite=True)


def _stream_args(directory, output, **overrides):
    old_argv = sys.argv
    sys.argv = ['originstack.py', '-d', str(directory), '-o', str(output),
                '--preset', 'quick', '--stream']
    try:
        from src.cli import apply_preset, parse_args
        args = parse_args()
    finally:
        sys.argv = old_argv
    apply_preset(args)
    if args.debayer_method == 'bilinear':
        args.debayer_method = 'malvar'
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def _masters():
    return {'bias': None, 'dark': None, 'flat': None}


def test_survey_applies_hard_limit_gate_only(tmp_path):
    for i in range(5):
        _write_star_frame(str(tmp_path / f'L{i:04d}.fits'), seed=i)
    # Real noise (non-flat, passes the loader's validity check) but no
    # stars/nebula structure -- hard-limit's "no stars detected" check
    # should reject it regardless of the (skipped) statistical stage.
    starless_path = str(tmp_path / 'L9999.fits')
    from astropy.io import fits
    rng = np.random.default_rng(42)
    noise = 1000.0 + rng.standard_normal((160, 160)).astype(np.float32) * 5.0
    fits.PrimaryHDU(data=np.stack([noise, noise, noise], axis=0)).writeto(
        starless_path, overwrite=True)

    args = _stream_args(tmp_path, tmp_path / 'out.fits')
    with _MemmapManager() as mm:
        records, rgb_cache = survey(args, str(tmp_path), _masters(), mm)

        assert len(records) == 6
        starless_rec = [r for r in records if r.path == starless_path][0]
        assert starless_rec.accepted is False
        assert starless_rec.metrics.get('star_count', 0) == 0
        assert sum(1 for r in records if r.accepted) == 5


def test_survey_discards_full_res_data(tmp_path):
    _write_star_frame(str(tmp_path / 'L0.fits'), seed=0)
    args = _stream_args(tmp_path, tmp_path / 'out.fits')
    with _MemmapManager() as mm:
        records, rgb_cache = survey(args, str(tmp_path), _masters(), mm)
        assert len(records) == 1
        # FrameRecord has no pixel-array field at all -- discarding is
        # structural, not just "happens to be None". The calibrated pixel
        # data lives on disk (rgb_cache), addressed by cache_index.
        assert not hasattr(records[0], 'rgb')
        assert not hasattr(records[0], 'lum')
        assert isinstance(records[0].metrics, dict)
        assert 'score' in records[0].metrics
        assert records[0].cache_index == 0
        assert rgb_cache is not None
        assert rgb_cache.shape[0] == 1


def test_auto_advisor_noop_when_auto_not_set(tmp_path, monkeypatch):
    """--auto not passed: must not touch args at all (existing behavior)."""
    import src.auto_settings as auto_mod
    from src.stream_stack import _run_auto_advisor_for_stream

    calls = []
    monkeypatch.setattr(auto_mod, 'apply_auto_settings',
                        lambda *a, **kw: calls.append((a, kw)) or
                        ('unknown', 'Unknown', {}, [], {}))

    _write_star_frame(str(tmp_path / 'L0.fits'), seed=0)
    args = _stream_args(tmp_path, tmp_path / 'out.fits')
    args.auto = False
    with _MemmapManager() as mm:
        records, _ = survey(args, str(tmp_path), _masters(), mm)
    _run_auto_advisor_for_stream(records, args, str(tmp_path))
    assert calls == []


def test_auto_advisor_applies_settings_when_auto_set(tmp_path, monkeypatch):
    """--auto passed: apply_auto_settings must be invoked with the accepted
    frames' metrics, and its returned changes must land on args (proven via
    a stub that flips a real args attribute, avoiding a dependency on the
    real pixel-signal classifier's behavior on tiny synthetic data)."""
    import src.auto_settings as auto_mod
    from src.stream_stack import _run_auto_advisor_for_stream

    def _stub(final, args, prior_type=None, prior_confidence=0.0):
        assert len(final) >= 1
        assert final[0].metrics is not None
        args.deconvolve = False
        args.ghs_b = 42.0
        return 'emission_nebula', 'Emission Nebula', {'n_frames': len(final)}, [
            'deconvolve  True -> False', 'ghs_b  8.0 -> 42.0'], {}

    monkeypatch.setattr(auto_mod, 'apply_auto_settings', _stub)

    for i in range(3):
        _write_star_frame(str(tmp_path / f'L{i:04d}.fits'), seed=i)
    args = _stream_args(tmp_path, tmp_path / 'out.fits')
    args.auto = True
    args.deconvolve = True
    args.ghs_b = 8.0
    with _MemmapManager() as mm:
        records, _ = survey(args, str(tmp_path), _masters(), mm)
    _run_auto_advisor_for_stream(records, args, str(tmp_path))

    assert args.deconvolve is False
    assert args.ghs_b == 42.0


def test_select_reference_picks_highest_score():
    good = FrameRecord(path='good.fits', header={}, metrics={'score': 90.0}, accepted=True)
    bad = FrameRecord(path='bad.fits', header={}, metrics={'score': 10.0}, accepted=True)
    rejected = FrameRecord(path='rejected.fits', header={}, metrics={'score': 99.0}, accepted=False)
    ref = select_reference([bad, good, rejected])
    assert ref is good


def test_select_reference_raises_with_no_accepted_frames():
    rejected = FrameRecord(path='r.fits', header={}, metrics={'score': 50.0}, accepted=False)
    with pytest.raises(ValueError):
        select_reference([rejected])


def test_fold_produces_registered_stack(tmp_path):
    shifts = [(0, 0), (3, -2), (-4, 5), (2, 6), (-3, -5)]
    for i, s in enumerate(shifts):
        _write_star_frame(str(tmp_path / f'L{i:04d}.fits'), shift=s, seed=i)

    args = _stream_args(tmp_path, tmp_path / 'out.fits')
    with _MemmapManager() as mm:
        records, rgb_cache = survey(args, str(tmp_path), _masters(), mm)
        reference = select_reference(records)
        stacked, frame_infos, out_shifts, total_exp, n_rej = fold(
            args, reference, records, _masters(), burn_in=3, sigma=3.0,
            rgb_cache=rgb_cache)

        assert len(frame_infos) == len(shifts)
        assert total_exp == pytest.approx(len(shifts) * 30.0)
        lum = 0.299 * stacked[:, :, 0] + 0.587 * stacked[:, :, 1] + 0.114 * stacked[:, :, 2]
        # Registration coherence: aligned stars stay sharp (a mis-stack smears
        # the peak toward the background level).
        assert lum.max() > 1000.0 + 1500.0


def test_fold_rejects_outlier_pixels(tmp_path):
    """One frame gets a bright injected hit (simulated cosmic ray/satellite
    trail) at a fixed pixel; the streamed sigma-clip fold should suppress it
    far more than a plain unrejected mean would."""
    n_frames = 14
    hit_yx = (10, 10)  # away from the synthetic nebula peak (centered at H/2,W/2)
                       # and star positions (placed in [20, H-20)) -- pure ~1000
                       # ADU background + noise there, so a hardcoded baseline is valid.
    for i in range(n_frames):
        hit = (hit_yx[0], hit_yx[1], 8000.0) if i == 5 else None
        _write_star_frame(str(tmp_path / f'L{i:04d}.fits'), seed=i, hit=hit)

    args = _stream_args(tmp_path, tmp_path / 'out.fits')
    with _MemmapManager() as mm:
        records, rgb_cache = survey(args, str(tmp_path), _masters(), mm)
        reference = select_reference(records)
        stacked, frame_infos, shifts, total_exp, n_rej = fold(
            args, reference, records, _masters(), burn_in=10, sigma=3.0,
            rgb_cache=rgb_cache)

        assert len(frame_infos) == n_frames
        # Plain mean would show ~8000/14 ~= 571 ADU excess at the hit pixel
        # over the ~1000 ADU background; sigma-clip should suppress almost
        # all of it.
        excess = float(stacked[hit_yx[0], hit_yx[1], 0]) - 1000.0
        assert excess < 200.0
        assert n_rej > 0


def test_fold_uses_disk_cache_not_reload(tmp_path, monkeypatch):
    """Once a frame's calibrated RGB is cached during survey, fold() must
    not call _process_single_frame for it again -- the whole point of the
    disk-cache middle ground (avoid re-load/re-calibrate/re-debayer)."""
    for i in range(5):
        _write_star_frame(str(tmp_path / f'L{i:04d}.fits'), seed=i)

    args = _stream_args(tmp_path, tmp_path / 'out.fits')
    with _MemmapManager() as mm:
        records, rgb_cache = survey(args, str(tmp_path), _masters(), mm)
        assert all(r.cache_index is not None for r in records)  # uniform shape -> all cached
        reference = select_reference(records)

        import src.frame_processor as fp_mod
        original = fp_mod._process_single_frame
        calls = []

        def _counting(*a, **kw):
            calls.append(a[0])
            return original(*a, **kw)

        monkeypatch.setattr(fp_mod, '_process_single_frame', _counting)

        stacked, frame_infos, shifts, total_exp, n_rej = fold(
            args, reference, records, _masters(), burn_in=3, sigma=3.0,
            rgb_cache=rgb_cache)

        assert len(calls) == 0  # fully served from cache, zero reloads
        assert len(frame_infos) == 5


def test_fold_falls_back_to_reload_without_cache(tmp_path):
    """rgb_cache=None (or a record with cache_index=None) must still work
    correctly -- the pre-caching fallback path stays intact."""
    for i in range(5):
        _write_star_frame(str(tmp_path / f'L{i:04d}.fits'), seed=i)

    args = _stream_args(tmp_path, tmp_path / 'out.fits')
    with _MemmapManager() as mm:
        records, rgb_cache = survey(args, str(tmp_path), _masters(), mm)
        reference = select_reference(records)
        # Force every record to look uncached.
        for r in records:
            r.cache_index = None

        stacked, frame_infos, shifts, total_exp, n_rej = fold(
            args, reference, records, _masters(), burn_in=3, sigma=3.0,
            rgb_cache=rgb_cache)

        assert len(frame_infos) == 5


def test_run_stream_stack_writes_fits_and_preview(tmp_path):
    shifts = [(0, 0), (2, -1), (-1, 2), (1, 1), (-2, -2), (0, 3)]
    for i, s in enumerate(shifts):
        _write_star_frame(str(tmp_path / f'L{i:04d}.fits'), shift=s, seed=i)

    out_path = tmp_path / 'out.fits'
    args = _stream_args(tmp_path, out_path, stream_burnin=3, output_tiff=True)
    rc = run_stream_stack(args)
    assert rc == 0
    assert out_path.exists()

    jpg_path = out_path.with_suffix('.jpg')
    assert jpg_path.exists()

    tiff_path = out_path.with_suffix('.tiff')
    assert tiff_path.exists()

    from astropy.io import fits
    with fits.open(str(out_path)) as hdul:
        hdr = hdul[0].header
        assert hdr['NFRAMES'] == len(shifts)
        assert hdr['COMBINED'] is True
        assert hdr['STREAMED'] is True
        assert hdr['RAWSTACK'] is True  # FITS holds the linear pre-Phase-4 stack
        assert hdul[0].data.shape[0] == 3  # (C,H,W) on disk


def test_stream_and_live_are_mutually_exclusive(tmp_path):
    old_argv = sys.argv
    sys.argv = ['originstack.py', '-d', str(tmp_path), '--live', '--stream']
    try:
        from src.cli import main
        with pytest.raises(SystemExit) as exc_info:
            main()
    finally:
        sys.argv = old_argv
    assert exc_info.value.code == 1
