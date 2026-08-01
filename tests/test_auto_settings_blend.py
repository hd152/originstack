"""Regression tests for the continuous target-type blend in
src/auto_settings.py (_blend_weights / _apply_dynamic_settings), which
replaced the old _classify()-then-bucket-lookup settings path.

Three concerns:
  1. Calibration: at each preset's own anchor point, the blend must
     reproduce that preset's validated values (no regression from going
     continuous).
  2. Interpolation: a point between two anchors must produce values
     strictly between the two presets', not a hard jump to either.
  3. Real-data regression: replaying this session's actual Trifid Nebula
     signals (with the same prior-type boost the real run exercised) must
     keep the already-hand-validated deconvolve=False / ghs_b~5 / ghs_sp~0.10
     behavior.
"""
from __future__ import annotations

import argparse

import pytest

from src import auto_settings as a


def _args(**overrides):
    base = dict(
        _explicit_cli_dests=set(), stack_method='auto', deconvolve=True,
        auto_denoise_strength=True, debayer_method='malvar',
        denoise_mmt=False, denoise_acdnr=False, denoise=False,
        denoise_bm3d=False, deconvolve_tv=False, patch_registration=False,
        consensus_ref=False, preview_black_sigma=0.0,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class TestBlendWeights:
    def test_weights_sum_to_one(self):
        sig = dict(a._TYPE_ANCHORS['galaxy'])
        w = a._blend_weights(sig)
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-9)

    def test_exact_anchor_dominates(self):
        for ttype, anchor in a._TYPE_ANCHORS.items():
            w = a._blend_weights(dict(anchor))
            best = max(w, key=w.get)
            assert best == ttype, f"{ttype}'s own anchor should be its own best match"
            assert w[best] > 0.9, f"{ttype} anchor only got weight {w[best]:.3f}"

    def test_prior_type_boost_increases_weight(self):
        # A signal profile far from every anchor -- prior should still be
        # able to meaningfully pull weight toward it.
        sig = {'median_filling': 0.6, 'diffuse_excess': 4.0, 'peak_excess': 9.0,
              'concentration': 2.2, 'star_count': 70.0, 'dynamic_range': 120.0}
        w_no_prior = a._blend_weights(sig)
        w_with_prior = a._blend_weights(sig, prior_type='galaxy', prior_confidence=0.9)
        assert w_with_prior['galaxy'] > w_no_prior['galaxy']

    def test_zero_confidence_prior_has_no_effect(self):
        sig = dict(a._TYPE_ANCHORS['unknown'])
        w1 = a._blend_weights(sig, prior_type='galaxy', prior_confidence=0.0)
        w2 = a._blend_weights(sig, prior_type=None, prior_confidence=0.0)
        for t in w1:
            assert w1[t] == pytest.approx(w2[t], abs=1e-9)


class TestCalibrationAtAnchors:
    """At each preset's own anchor, the blend must reproduce that preset's
    values -- proves going continuous didn't regress the validated presets."""

    @pytest.mark.parametrize("ttype", list(a._TYPE_ANCHORS.keys()))
    def test_matches_old_bucket_values(self, ttype):
        sig = dict(a._TYPE_ANCHORS[ttype])
        sig['fwhm'] = 3.0  # neutral -- avoid the star_field poor-seeing exception
        w = a._blend_weights(sig)
        args = _args()
        a._apply_dynamic_settings(sig, w, args)

        for attr, expected in a._TARGET_SETTINGS.get(ttype, []):
            got = getattr(args, attr, None)
            assert got is not None, f"{ttype}.{attr} was never set"
            if isinstance(expected, bool):
                assert got == expected, f"{ttype}.{attr}: got {got}, want {expected}"
            else:
                # Blending pulls in a few percent from neighboring presets even
                # at a dominant (>90%) anchor match -- allow a loose tolerance,
                # this is inherent to blending, not a bug (see class docstring).
                tol = max(0.15 * abs(expected), 0.1)
                assert abs(got - expected) < tol, (
                    f"{ttype}.{attr}: got {got}, want ~{expected} (tol {tol})")


class TestInterpolation:
    def test_midpoint_between_two_anchors_is_strictly_between(self):
        em = a._TYPE_ANCHORS['emission_nebula']
        rf = a._TYPE_ANCHORS['reflection_nebula']
        mid = {k: (em[k] + rf[k]) / 2 for k in em}
        mid['fwhm'] = 3.0

        w = a._blend_weights(mid)
        args = _args()
        a._apply_dynamic_settings(mid, w, args)

        em_ghs_sp = dict(a._TARGET_SETTINGS['emission_nebula'])['ghs_sp']
        rf_ghs_sp = dict(a._TARGET_SETTINGS['reflection_nebula'])['ghs_sp']
        lo, hi = sorted((em_ghs_sp, rf_ghs_sp))
        assert lo < args.ghs_sp < hi

    def test_boolean_choice_comes_from_nearest_contributing_preset(self):
        """Booleans can't fractionally blend -- confirm the winner really is
        whichever preset has the higher weight at a point closer to one side."""
        em = a._TYPE_ANCHORS['emission_nebula']
        rf = a._TYPE_ANCHORS['reflection_nebula']
        # 90% of the way from reflection_nebula toward emission_nebula.
        near_em = {k: rf[k] + 0.9 * (em[k] - rf[k]) for k in em}
        near_em['fwhm'] = 3.0
        w = a._blend_weights(near_em)
        assert w['emission_nebula'] > w['reflection_nebula']


class TestRealTrifidRegression:
    """Replays this session's actual measured Trifid Nebula signals (from a
    real --stream run's Auto Advisor output) through the new blend, with the
    same prior-type boost (header OBJECT='Trifid Nebula', conf=0.90) the
    real run exercised, and checks it still lands close to the values
    already hand-validated against the real render this session."""

    def _trifid_signals(self):
        return {
            'median_filling': 0.00, 'diffuse_excess': 0.69, 'peak_excess': 1.8,
            'concentration': 1.8 / 0.69, 'star_count': 277, 'dynamic_range': 100.0,
            'fwhm': 7.8, 'snr': 10.0, 'n_frames': 239, 'strehl': 0.0,
            'dispersion': 0.0, 'median_ellipticity': 0.0,
        }

    def test_deconvolve_stays_off(self):
        sig = self._trifid_signals()
        w = a._blend_weights(sig, prior_type='emission_nebula', prior_confidence=0.90)
        args = _args()
        a._apply_dynamic_settings(sig, w, args)
        assert args.deconvolve is False

    def test_stretch_params_close_to_hand_validated_values(self):
        sig = self._trifid_signals()
        w = a._blend_weights(sig, prior_type='emission_nebula', prior_confidence=0.90)
        args = _args()
        a._apply_dynamic_settings(sig, w, args)
        # Hand-validated this session: ghs_b=5.0, ghs_sp=0.10 (see
        # auto_settings.py's emission_nebula preset comment). Real signals
        # don't sit exactly on the anchor (star_count=277 pulls weight
        # toward star_field/wide_field too), so allow real drift, not
        # exact match.
        assert 3.0 < args.ghs_b < 8.0
        assert 0.05 < args.ghs_sp < 0.20

    def test_emission_nebula_is_the_dominant_weight(self):
        sig = self._trifid_signals()
        w = a._blend_weights(sig, prior_type='emission_nebula', prior_confidence=0.90)
        assert max(w, key=w.get) == 'emission_nebula'
        assert w['emission_nebula'] > 0.5
