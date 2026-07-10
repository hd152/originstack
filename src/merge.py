"""Incremental stacking: merge previously saved linear stacks into this run.

A previous run's main output FITS is the linear, pre-post-processing stack
(``RAWSTACK=True``) with ``NFRAMES``/``TOTEXP``/date metadata in its header.
``--merge PREV.fits [...]`` registers each such stack onto the current
session's pixel grid (star-match affine with a translation fallback — nights
on an alt-az mount differ by arbitrary field rotation) and combines them as a
per-pixel weighted mean, weight = that stack's frame count inside its warped
footprint and 0 outside, so zero-filled borders never dilute the result:

    merged(x) = sum_i w_i(x) * S_i(x) / sum_i w_i(x)

There is no cross-session outlier rejection — each session already rejected
outliers internally when it was stacked (the same tradeoff hierarchical mode
makes). The merged output is itself a linear stack with summed headers, so
merges chain night after night.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.models import Config
from src.utils import safe_print, get_logger

_log = get_logger()


def _lum(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1]
            + 0.114 * rgb[:, :, 2]).astype(np.float32)


def _detect_stars(lum: np.ndarray) -> Optional[Any]:
    """Star catalog for affine matching; robust noise estimate from MAD."""
    from src.quality import _sep_detect_stars
    med = float(np.median(lum))
    noise = 1.4826 * float(np.median(np.abs(lum - med)))
    try:
        return _sep_detect_stars(lum, max(noise, 1e-3))
    except Exception:
        return None


def _embed_to_shape(arr: np.ndarray, H: int, W: int) -> np.ndarray:
    """Place ``arr`` in the top-left of an (H, W[, C]) zero canvas (crop if
    larger). Pixel coordinates are preserved, so a transform computed on the
    embedded luminance applies directly to the embedded image."""
    if arr.shape[0] == H and arr.shape[1] == W:
        return arr
    if arr.ndim == 3:
        out = np.zeros((H, W, arr.shape[2]), dtype=arr.dtype)
    else:
        out = np.zeros((H, W), dtype=arr.dtype)
    h = min(H, arr.shape[0])
    w = min(W, arr.shape[1])
    out[:h, :w] = arr[:h, :w]
    return out


def _register_stack(new_lum: np.ndarray, prev_lum: np.ndarray,
                    new_stars: Optional[Any]) -> Tuple[Optional[Any],
                                                       Optional[Tuple[float, float]]]:
    """Transform aligning ``prev`` onto ``new``: star-match affine first
    (rotation between nights is arbitrary), translation-only fallback.
    Returns (transform, shift) — exactly one is non-None on success,
    both None on failure."""
    from src.registration import (calculate_shift, match_stars_affine,
                                  _astroalign_transform)

    # astroalign first: its triangle-pattern matching handles the arbitrary
    # field rotation between nights. The nearest-neighbour+RANSAC star
    # matcher assumes rotation ~ 0 after translation (an in-session tool) and
    # can return a confidently wrong model on rotated fields.
    tf = _astroalign_transform(new_lum, prev_lum)
    if tf is not None:
        return tf, None

    shift = None
    try:
        sy, sx = calculate_shift(new_lum, prev_lum, skip_phase_cc=True,
                                 use_pyramid=True, masked_correlation=False,
                                 corr_downsample=2)
        if np.isfinite(sy) and np.isfinite(sx):
            shift = (float(sy), float(sx))
    except Exception as exc:
        _log.debug("merge: translation estimate failed: %s", exc)

    prev_stars = _detect_stars(prev_lum)
    if new_stars is not None and prev_stars is not None:
        tf = match_stars_affine(new_stars, prev_stars,
                                initial_shift=shift or (0.0, 0.0))
        if tf is not None:
            return tf, None
    if shift is not None:
        return None, shift
    return None, None


def load_merge_stack(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load and validate one ``--merge`` input. Returns (HWC float32, meta)."""
    from astropy.io import fits as _fits

    with _fits.open(path) as hdul:
        hdr = hdul[0].header
        data = np.asarray(hdul[0].data)
    if not bool(hdr.get('RAWSTACK', False)):
        raise ValueError(
            f"{os.path.basename(path)} is not a linear (pre-post-processing) "
            f"stack: header RAWSTACK is missing or False. Pass the main "
            f"output FITS of a previous run, not the _processed one.")
    if data.ndim == 3 and data.shape[0] == 3:
        data = np.moveaxis(data, 0, -1)
    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError(f"{os.path.basename(path)}: expected an RGB stack, "
                         f"got shape {data.shape}")
    meta = {
        'path': path,
        'nframes': int(hdr.get('NFRAMES', 0) or 0),
        'intgtime': float(hdr.get('INTGTIME', 0.0) or 0.0),
        'totexp': float(hdr.get('TOTEXP', 0.0) or 0.0),
        'datefrst': hdr.get('DATEFRST'),
        'datelast': hdr.get('DATELAST'),
    }
    return np.ascontiguousarray(data, dtype=np.float32), meta


def merge_previous_stacks(stacked: np.ndarray, new_frame_count: int,
                          merge_paths: List[str],
                          verbose: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Register and weighted-merge previous linear stacks into ``stacked``.

    Returns (merged HWC float32, merge_info). Raises ValueError on
    unusable inputs or failed registration — a silently wrong merge is worse
    than an error.
    """
    from src.registration import apply_transform

    H, W = stacked.shape[:2]
    new_w = float(max(new_frame_count, 1))

    acc = stacked.astype(np.float64) * new_w
    wsum = np.full((H, W), new_w, dtype=np.float64)

    new_lum = _lum(stacked)
    new_stars = _detect_stars(new_lum)

    info: Dict[str, Any] = {
        'n_sources': 0, 'sources': [],
        'total_frames': int(new_frame_count),
        'total_intgtime': 0.0, 'total_totexp': 0.0,
        'datefrst': None, 'datelast': None,
    }

    for path in merge_paths:
        prev, meta = load_merge_stack(path)
        w_prev = float(meta['nframes']) if meta['nframes'] > 0 else 1.0
        if meta['nframes'] <= 0:
            safe_print(f"  WARNING: {os.path.basename(path)} has no NFRAMES "
                       f"header — merging with weight 1.0")

        prev_h, prev_w = prev.shape[:2]
        prev = _embed_to_shape(prev, H, W)
        prev_lum = _lum(prev)
        embed_mask = _embed_to_shape(
            np.ones((prev_h, prev_w), dtype=np.float32), H, W)

        tf, shift = _register_stack(new_lum, prev_lum, new_stars)
        if tf is None and shift is None:
            raise ValueError(
                f"Could not register {os.path.basename(path)} to the current "
                f"stack (no star match and no correlation peak) — is it the "
                f"same target?")

        # Negligible transform (already-aligned grids, registration jitter
        # only): skip the warp — resampling costs more than a <0.05px /
        # <0.001deg correction is worth (same principle as the CA
        # sub-threshold skip).
        if shift is not None:
            negligible = max(abs(shift[0]), abs(shift[1])) < 0.05
        else:
            _rot = float(np.arctan2(tf.params[1, 0], tf.params[0, 0]))
            _t = np.abs(tf.params[:2, 2]).max()
            negligible = abs(_rot) < 2e-5 and _t < 0.05

        if negligible:
            aligned = prev
            footprint = embed_mask
        else:
            aligned = apply_transform(prev, shift=shift, transform=tf)
            footprint = apply_transform(embed_mask, shift=shift, transform=tf)
        valid = footprint > 0.5

        overlap = float(np.mean(valid))
        if overlap < Config.MERGE_MIN_OVERLAP:
            raise ValueError(
                f"{os.path.basename(path)} overlaps only {overlap * 100:.0f}% "
                f"of the current frame (< {Config.MERGE_MIN_OVERLAP * 100:.0f}%) "
                f"— refusing to merge (wrong target or failed registration?)")

        # Post-alignment sanity: the aligned stack must actually look like
        # the new one (same stars). Catches wrong-target inputs and silently
        # failed registrations that still produced a plausible footprint.
        al = _lum(aligned)[::4, ::4]
        nl = new_lum[::4, ::4]
        vm = valid[::4, ::4]
        if vm.sum() > 100:
            a_v, n_v = al[vm].astype(np.float64), nl[vm].astype(np.float64)
            if a_v.std() > 1e-9 and n_v.std() > 1e-9:
                corr = float(np.corrcoef(a_v, n_v)[0, 1])
            else:
                corr = 0.0
            if corr < Config.MERGE_MIN_CORRELATION:
                raise ValueError(
                    f"{os.path.basename(path)}: aligned stack correlates only "
                    f"{corr:.2f} with the current stack "
                    f"(< {Config.MERGE_MIN_CORRELATION}) — wrong target or "
                    f"failed registration; refusing to merge")

        w_map = np.where(valid, w_prev, 0.0)
        acc += aligned.astype(np.float64) * w_map[:, :, np.newaxis]
        wsum += w_map

        if tf is not None:
            t_xy = tf.params[:2, 2]
            sy, sx = float(t_xy[1]), float(t_xy[0])
            rot = float(np.degrees(np.arctan2(tf.params[1, 0], tf.params[0, 0])))
        else:
            sy, sx = shift
            rot = 0.0
        safe_print(f"  Merged {os.path.basename(path)}: {meta['nframes']} frames, "
                   f"shift=({sx:+.1f}, {sy:+.1f})px rot={rot:+.2f}deg, "
                   f"overlap {overlap * 100:.0f}%")

        info['n_sources'] += 1
        info['sources'].append(os.path.basename(path))
        info['total_frames'] += meta['nframes']
        info['total_intgtime'] += meta['intgtime']
        info['total_totexp'] += meta['totexp']
        for key, pick in (('datefrst', min), ('datelast', max)):
            v = meta[key]
            if v:
                info[key] = pick(info[key], str(v)) if info[key] else str(v)

    merged = (acc / np.maximum(wsum[:, :, np.newaxis], 1e-12)).astype(np.float32)
    return merged, info


def apply_merge_header(header, info: Dict[str, Any]) -> None:
    """Override the session header aggregates with merged totals."""
    header['NFRAMES'] = (info['total_frames'], 'Number of stacked frames (all merged sessions)')
    header['MERGED'] = (info['n_sources'], 'Number of previous stacks merged in')
    for i, src in enumerate(info['sources'][:20]):
        header[f'MRGSRC{i + 1}'] = (src[:68], 'Merged source stack')
    if info['total_intgtime'] > 0 and 'INTGTIME' in header:
        total = float(header['INTGTIME']) + info['total_intgtime']
        header['INTGTIME'] = (total, 'Total integration time across all frames (seconds)')
        if 'INTGMIN' in header:
            header['INTGMIN'] = (total / 60.0, 'Total integration time (minutes)')
    if info['total_totexp'] > 0 and 'TOTEXP' in header:
        header['TOTEXP'] = (float(header['TOTEXP']) + info['total_totexp'],
                            'Total integrated exposure time in seconds')
    if info['datefrst'] and 'DATEFRST' in header:
        header['DATEFRST'] = min(str(header['DATEFRST']), info['datefrst'])
    if info['datelast'] and 'DATELAST' in header:
        header['DATELAST'] = max(str(header['DATELAST']), info['datelast'])
