"""Mosaic detection (``predict_mosaic``) and stitching (``stitch_mosaic_panels``) on
synthetic panels with known truth."""
import json
import math
import types

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from src import mosaic

FOV_X, FOV_Y = 0.02189, 0.01467          # radians, the Origin's field (info.json fovX/fovY)


def _session(tmp_path, name, ra_deg, dec_deg, extra=None):
    d = tmp_path / name
    d.mkdir()
    si = {"celestial": {"first": math.radians(ra_deg), "second": math.radians(dec_deg)},
          "fovX": FOV_X, "fovY": FOV_Y, "objectName": "M31 mosaic"}
    if extra:
        si.update(extra)
    (d / "info.json").write_text(json.dumps({"StackedInfo": si}))
    return str(d)


def test_tiles_at_overlapping_pointings_are_a_mosaic(tmp_path):
    step = math.degrees(FOV_Y) * 0.8                  # 20% overlap in the short direction
    dirs = [_session(tmp_path, f"t{i}", 10.0, 41.0 + i * step) for i in range(3)]
    got = mosaic.predict_mosaic(dirs)
    assert got is not None and len(got["clusters"]) == 3


def test_several_nights_of_one_tile_stay_one_panel(tmp_path):
    step = math.degrees(FOV_Y) * 0.8
    dirs = [_session(tmp_path, "a1", 10.0, 41.0), _session(tmp_path, "a2", 10.003, 41.002),
            _session(tmp_path, "b1", 10.0, 41.0 + step)]
    got = mosaic.predict_mosaic(dirs)
    assert got is not None
    assert sorted(len(c) for c in got["clusters"]) == [1, 2]


def test_nights_of_the_same_target_are_not_a_mosaic(tmp_path):
    # real nights of one target differ by 7-70 px of a 2048 px field
    dirs = [_session(tmp_path, f"n{i}", 202.47 + 0.004 * i, 47.19 - 0.003 * i) for i in range(4)]
    assert mosaic.predict_mosaic(dirs) is None


def test_unrelated_targets_are_not_a_mosaic(tmp_path):
    dirs = [_session(tmp_path, "m51", 202.47, 47.19), _session(tmp_path, "m101", 210.8, 54.35)]
    assert mosaic.predict_mosaic(dirs) is None


def test_missing_metadata_is_not_a_guess(tmp_path):
    d1 = _session(tmp_path, "t0", 10.0, 41.0)
    d2 = tmp_path / "t1"
    d2.mkdir()
    assert mosaic.predict_mosaic([d1, str(d2)]) is None


def test_origin_tile_keys_are_reported(tmp_path):
    step = math.degrees(FOV_Y) * 0.8
    dirs = [_session(tmp_path, f"t{i}", 10.0, 41.0 + i * step,
                     {"MosaicUuid": "abc", "TileRow": i, "TileCol": 0}) for i in range(2)]
    got = mosaic.predict_mosaic(dirs)
    assert got is not None and "tile keys" in got["reason"]
    assert mosaic.session_pointing(dirs[1])["tile"]["tilerow"] == 1


# ── stitching ──────────────────────────────────────────────────────────────────

def _sky(H, W, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    img = np.full((H, W, 3), 100.0, np.float32)
    img += (300.0 * np.exp(-(((yy - H * 0.55) / 90.0) ** 2 + ((xx - W * 0.45) / 160.0) ** 2)))[..., None] \
        * np.array([1.0, 0.8, 0.6], np.float32)
    for _ in range(400):
        cy, cx = rng.uniform(0, H), rng.uniform(0, W)
        f = rng.uniform(50, 2000)
        r2 = (yy - cy) ** 2 + (xx - cx) ** 2
        img += (f * np.exp(-r2 / (2 * 1.6 ** 2)))[..., None] * rng.uniform(0.7, 1.3, 3).astype(np.float32)
    return img


def _wcs(ra, dec, scale_deg, crpix, rot_deg=0.0):
    """TAN WCS with RA increasing with +x (as real Origin frames), rotated by rot_deg."""
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra, dec]
    c, s = math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg))
    w.wcs.cd = scale_deg * np.array([[c, -s], [s, c]])
    w.wcs.crpix = list(crpix)
    return w


def _write_panels(tmp_path, truth, w_truth, boxes, gains, planes, sigma, rng):
    paths = []
    H, W = truth.shape[:2]
    for k, ((y0, y1, x0, x1), g, (a, b, c)) in enumerate(zip(boxes, gains, planes)):
        sub = truth[y0:y1, x0:x1].copy()
        h, w = sub.shape[:2]
        v, u = np.mgrid[:h, :w].astype(np.float32)
        grad = a + b * (u / w - 0.5) + c * (v / h - 0.5)
        data = (sub / g) + grad[..., None] + rng.normal(0, sigma, sub.shape).astype(np.float32)
        w = w_truth.deepcopy()
        w.wcs.crpix = [w_truth.wcs.crpix[0] - x0, w_truth.wcs.crpix[1] - y0]
        hdr = w.to_header()
        hdr["RAWSTACK"] = True
        hdr["NFRAMES"] = 100 + k
        hdr["INTGTIME"] = float(2000 + 100 * k)
        p = tmp_path / f"panel{k}.fits"
        fits.PrimaryHDU(np.transpose(data, (2, 0, 1)).astype(np.float32), hdr).writeto(p)
        paths.append(str(p))
    return paths


# 7 deg: field rotation as on real Origin stacks (their CD is rotated ~6.9 deg)
@pytest.mark.parametrize("rot", [0.0, 7.0])
def test_stitch_recovers_the_sky_without_seams(tmp_path, rot):
    rng = np.random.default_rng(3)
    H, W = 700, 1000
    truth = _sky(H, W)
    w_truth = _wcs(10.0, 41.0, 1.476 / 3600, (W / 2, H / 2), rot)
    boxes = [(0, 420, 0, 600), (0, 420, 400, 1000), (280, 700, 0, 600), (280, 700, 400, 1000)]
    gains = [1.0, 1.08, 0.93, 1.03]
    planes = [(0, 0, 0), (12.0, 8.0, -5.0), (-7.0, -6.0, 10.0), (4.0, 10.0, 6.0)]
    sigma = 2.0
    paths = _write_panels(tmp_path, truth, w_truth, boxes, gains, planes, sigma, rng)
    out = tmp_path / "mosaic.fits"
    assert mosaic.stitch_mosaic_panels(paths, str(out), types.SimpleNamespace(verbose=False))

    with fits.open(out) as hd:
        m = np.moveaxis(hd[0].data, 0, -1)
        hdr = hd[0].header
    assert hdr["RAWSTACK"] and hdr["MOSAIC"] and hdr["NPANELS"] == 4
    # the four panels tile the whole field: nothing may fall off the grid
    assert hdr["COVERAGE"] > 0.95, hdr["COVERAGE"]
    assert abs(m.shape[0] - H) <= 2 and abs(m.shape[1] - W) <= 2
    w_m = WCS(hdr, naxis=2)
    # sample the truth at the mosaic's pixels (exact: same scale, axis-aligned, integer shift)
    ra, dec = w_m.pixel_to_world_values(0, 0)
    tx, ty = w_truth.world_to_pixel_values(ra, dec)
    ox, oy = int(round(float(tx))), int(round(float(ty)))
    my, mx = max(0, -oy), max(0, -ox)                       # mosaic rows/cols before the truth's
    ty0, tx0 = oy + my, ox + mx
    hh, ww = min(m.shape[0] - my, H - ty0), min(m.shape[1] - mx, W - tx0)
    got = m[my:my + hh, mx:mx + ww]
    ref = truth[ty0:ty0 + hh, tx0:tx0 + ww] / gains[3]      # deepest panel (k=3) is the scale
    inner = np.zeros((hh, ww), bool)
    inner[10:-10, 10:-10] = True
    # residual vs truth: an offset/plane from the reference panel is allowed; remove it
    res = (got - ref)[inner]
    res -= np.median(res, axis=0)
    rms = np.sqrt(np.mean(np.clip(res, -50, 50) ** 2, axis=0))
    assert np.all(rms < 3.0 * sigma), rms
    # no seams: after one global plane (the deepest panel's own gradient, which the
    # stitch keeps by design), every 16 px block of the residual is ~0 -- including the
    # overlap strips, where an unmatched gain or gradient would show
    for c in range(3):
        d = (got - ref)[:hh - hh % 16, :ww - ww % 16, c]
        blk = np.median(d.reshape(hh // 16, 16, ww // 16, 16).transpose(0, 2, 1, 3).reshape(hh // 16, ww // 16, -1), -1)
        by, bx = np.mgrid[:blk.shape[0], :blk.shape[1]]
        A = np.column_stack([np.ones(blk.size), bx.ravel(), by.ravel()])
        coef, *_ = np.linalg.lstsq(A, blk.ravel(), rcond=None)
        left = (blk.ravel() - A @ coef)[1:-1]
        assert np.percentile(np.abs(left), 99) < 1.0 * sigma, (c, np.percentile(np.abs(left), 99))


def test_panels_that_do_not_overlap_are_refused(tmp_path):
    rng = np.random.default_rng(4)
    H, W = 400, 1200
    truth = _sky(H, W, seed=1)
    w_truth = _wcs(10.0, 41.0, 1.476 / 3600, (W / 2, H / 2))
    boxes = [(0, 400, 0, 500), (0, 400, 700, 1200)]
    paths = _write_panels(tmp_path, truth, w_truth, boxes, [1.0, 1.0], [(0, 0, 0)] * 2, 2.0, rng)
    assert not mosaic.stitch_mosaic_panels(paths, str(tmp_path / "m.fits"),
                                           types.SimpleNamespace(verbose=False))


def test_a_panel_without_wcs_is_refused(tmp_path):
    p = tmp_path / "nowcs.fits"
    fits.PrimaryHDU(np.zeros((3, 64, 64), np.float32)).writeto(p)
    assert not mosaic.stitch_mosaic_panels([str(p), str(p)], str(tmp_path / "m.fits"),
                                           types.SimpleNamespace(verbose=False))


@pytest.mark.parametrize("flag", ["combine_sessions", "hierarchical"])
def test_explicit_session_flags_override_detection(tmp_path, flag):
    from src.cli import _want_combine_sessions
    step = math.degrees(FOV_Y) * 0.8
    dirs = [_session(tmp_path, f"t{i}", 10.0, 41.0 + i * step) for i in range(2)]
    args = types.SimpleNamespace(mosaic=False, combine_sessions=False, hierarchical=False)
    setattr(args, flag, True)
    _want_combine_sessions(args, dirs)
    assert args.mosaic is False


def test_detected_mosaic_switches_to_panels(tmp_path):
    from src.cli import _want_combine_sessions
    step = math.degrees(FOV_Y) * 0.8
    dirs = [_session(tmp_path, f"t{i}", 10.0, 41.0 + i * step) for i in range(2)]
    args = types.SimpleNamespace(mosaic=False, combine_sessions=False, hierarchical=False)
    assert _want_combine_sessions(args, dirs) is False
    assert args.mosaic is True
