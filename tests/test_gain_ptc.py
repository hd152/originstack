"""Tests for src/gain_ptc.py -- photon-transfer gain / read-noise estimate."""
from __future__ import annotations

import numpy as np
import pytest

from src.gain_ptc import estimate_gain_ptc


def _synthetic_calibration(gain, read_noise_e, seed=3, H=300, W=320,
                           sig_adu=5000.0, bias_off=100.0):
    rng = np.random.default_rng(seed)
    rn_adu = read_noise_e / gain

    def bias():
        return (bias_off + rng.normal(0.0, rn_adu, (H, W))).astype(np.float32)

    def flat():
        shot_adu = np.sqrt(sig_adu * gain) / gain          # sqrt(N_e)/G
        total_adu = np.sqrt(shot_adu ** 2 + rn_adu ** 2)
        # smooth vignetting -- must cancel in the frame difference
        vig = 1.0 - 0.12 * (np.linspace(-1, 1, W)[None, :] ** 2)
        return (bias_off + sig_adu * vig
                + rng.normal(0.0, total_adu, (H, W))).astype(np.float32)

    store = {"b0": bias(), "b1": bias(), "f0": flat(), "f1": flat()}
    return lambda path: (store[path], {})


def test_recovers_known_gain_and_read_noise():
    load = _synthetic_calibration(gain=2.0, read_noise_e=8.0)
    g, rn = estimate_gain_ptc(["b0", "b1"], ["f0", "f1"], load_fn=load)
    assert g == pytest.approx(2.0, rel=0.08)
    assert rn == pytest.approx(8.0, rel=0.25)


def test_recovers_a_different_gain():
    load = _synthetic_calibration(gain=0.5, read_noise_e=4.0, seed=11)
    g, rn = estimate_gain_ptc(["b0", "b1"], ["f0", "f1"], load_fn=load)
    assert g == pytest.approx(0.5, rel=0.1)


def test_insufficient_frames_returns_none():
    load = _synthetic_calibration(gain=2.0, read_noise_e=8.0)
    assert estimate_gain_ptc(["b0"], ["f0", "f1"], load_fn=load) == (None, None)
    assert estimate_gain_ptc(["b0", "b1"], ["f0"], load_fn=load) == (None, None)


def test_flat_at_bias_level_is_rejected():
    # "flats" that carry no real signal above the bias -> no PTC solution.
    rng = np.random.default_rng(1)
    store = {k: (100.0 + rng.normal(0, 3.0, (120, 120))).astype(np.float32)
             for k in ("b0", "b1", "f0", "f1")}
    load = lambda p: (store[p], {})
    assert estimate_gain_ptc(["b0", "b1"], ["f0", "f1"], load_fn=load) == (None, None)
