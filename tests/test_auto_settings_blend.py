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
from types import SimpleNamespace

import pytest

from src import auto_settings as a


def _args(**overrides):
    base = dict(
        _explicit_cli_dests=set(), stack_method='auto', deconvolve=True,
        auto_denoise_strength=True, debayer_method='malvar',
        denoise_mmt=False, denoise_acdnr=False, denoise=False,
        denoise_bm3d=False, deconvolve_tv=False, patch_registration=False,
        consensus_ref=False, preview_black_sigma=0.0,
        drizzle_scale=1.0, elastic_registration=False,
        drizzle_kernel='lanczos3', super_res_iters=0, denoise_gain=None,
        gaia_distortion_correction=False,
        pre_gradient_removal=False, dbe_patch_size=64, bg_clip_sigma=3.0,
        local_normalize=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _sig(n_frames=10, star_count=10, snr=10.0, fwhm=3.0, strehl=0.0, dispersion=0.0,
        median_ellipticity=0.0, **overrides):
    base = dict(n_frames=n_frames, star_count=star_count, snr=snr, fwhm=fwhm,
               strehl=strehl, dispersion=dispersion, median_ellipticity=median_ellipticity,
               median_filling=0.0, diffuse_excess=0.0, peak_excess=0.0,
               concentration=0.0, dynamic_range=0.0)
    base.update(overrides)
    return base


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


class TestWaveletCombineAutoRule:
    """--auto folding: stack_method='wavelet' (src/auto_settings.py rule 1
    extension) for fine-structure-heavy targets in the right frame-count
    band, no drizzle requested."""

    def _fine_structure_weights(self):
        return {'emission_nebula': 0.4, 'reflection_nebula': 0.2, 'galaxy': 0.1,
               'star_field': 0.1, 'wide_field': 0.1, 'globular_cluster': 0.05,
               'planetary_nebula': 0.03, 'unknown': 0.02}

    def test_selects_wavelet_in_band(self):
        sig = _sig(n_frames=15)
        args = _args()
        a._apply_quality_settings(sig, args, weights=self._fine_structure_weights())
        assert args.stack_method == 'wavelet'

    def test_skips_wavelet_when_drizzling(self):
        sig = _sig(n_frames=15)
        args = _args(drizzle_scale=2.0)
        a._apply_quality_settings(sig, args, weights=self._fine_structure_weights())
        assert args.stack_method != 'wavelet'

    def test_skips_wavelet_below_frame_floor(self):
        sig = _sig(n_frames=5)
        args = _args()
        a._apply_quality_settings(sig, args, weights=self._fine_structure_weights())
        assert args.stack_method != 'wavelet'

    def test_skips_wavelet_above_frame_ceiling(self):
        sig = _sig(n_frames=80)
        args = _args()
        a._apply_quality_settings(sig, args, weights=self._fine_structure_weights())
        assert args.stack_method != 'wavelet'

    def test_skips_wavelet_for_star_field(self):
        star_field_weights = {'star_field': 0.9, 'wide_field': 0.1}
        sig = _sig(n_frames=15)
        args = _args()
        a._apply_quality_settings(sig, args, weights=star_field_weights)
        assert args.stack_method != 'wavelet'

    def test_explicit_stack_method_wins(self):
        sig = _sig(n_frames=15)
        args = _args(stack_method='sigma_clip', _explicit_cli_dests={'stack_method'})
        a._apply_quality_settings(sig, args, weights=self._fine_structure_weights())
        assert args.stack_method == 'sigma_clip'


class TestElasticRegistrationAutoRule:
    """--auto folding: elastic_registration=True, method='star' for
    high-frame-count, star-rich sessions only."""

    def test_enables_star_method_when_thresholds_met(self):
        sig = _sig(n_frames=25, star_count=40)
        args = _args()
        a._apply_quality_settings(sig, args, weights={})
        assert args.elastic_registration is True
        assert args.elastic_registration_method == 'star'

    def test_stays_off_below_frame_floor(self):
        sig = _sig(n_frames=15, star_count=40)
        args = _args()
        a._apply_quality_settings(sig, args, weights={})
        assert args.elastic_registration is False

    def test_stays_off_below_star_floor(self):
        sig = _sig(n_frames=25, star_count=10)
        args = _args()
        a._apply_quality_settings(sig, args, weights={})
        assert args.elastic_registration is False

    def test_explicit_flag_not_overridden(self):
        sig = _sig(n_frames=25, star_count=40)
        args = _args(elastic_registration=False,
                     _explicit_cli_dests={'elastic_registration'})
        a._apply_quality_settings(sig, args, weights={})
        assert args.elastic_registration is False


class TestDrizzlePsfAndIbpAutoRule:
    """--auto folding: drizzle_kernel='psf' + super_res_iters=5, only when
    the user already requested --drizzle-scale > 1 and elastic registration
    isn't active."""

    def test_enables_when_drizzling(self):
        sig = _sig()
        args = _args(drizzle_scale=2.0)
        a._apply_quality_settings(sig, args, weights={})
        assert args.drizzle_kernel == 'psf'
        assert args.super_res_iters == 5

    def test_no_change_without_drizzle(self):
        sig = _sig()
        args = _args(drizzle_scale=1.0)
        a._apply_quality_settings(sig, args, weights={})
        assert args.drizzle_kernel == 'lanczos3'
        assert args.super_res_iters == 0

    def test_skipped_when_elastic_registration_active(self):
        sig = _sig()
        args = _args(drizzle_scale=2.0, elastic_registration=True)
        a._apply_quality_settings(sig, args, weights={})
        assert args.drizzle_kernel == 'lanczos3'
        assert args.super_res_iters == 0

    def test_explicit_drizzle_kernel_not_overridden(self):
        sig = _sig()
        args = _args(drizzle_scale=2.0, drizzle_kernel='lanczos3',
                     _explicit_cli_dests={'drizzle_kernel'})
        a._apply_quality_settings(sig, args, weights={})
        assert args.drizzle_kernel == 'lanczos3'


class TestDenoiseGainEgainAutoRule:
    """--auto folding: denoise_gain set only from a literal EGAIN header key
    (never from GAIN, which is a camera setting, not e-/ADU)."""

    def test_sets_from_egain_header(self):
        sig = _sig()
        args = _args()
        final = [SimpleNamespace(header={'EGAIN': 1.2}, metrics={})]
        a._apply_quality_settings(sig, args, weights={}, final=final)
        assert args.denoise_gain == pytest.approx(1.2)

    def test_ignores_bare_gain_header(self):
        sig = _sig()
        args = _args()
        final = [SimpleNamespace(header={'GAIN': 200}, metrics={})]  # camera setting, not e-/ADU
        a._apply_quality_settings(sig, args, weights={}, final=final)
        assert args.denoise_gain is None

    def test_ignores_out_of_range_egain(self):
        sig = _sig()
        args = _args()
        final = [SimpleNamespace(header={'EGAIN': 500.0}, metrics={})]
        a._apply_quality_settings(sig, args, weights={}, final=final)
        assert args.denoise_gain is None

    def test_no_final_no_crash(self):
        sig = _sig()
        args = _args()
        a._apply_quality_settings(sig, args, weights={}, final=None)
        assert args.denoise_gain is None

    def test_explicit_denoise_gain_not_overridden(self):
        sig = _sig()
        args = _args(denoise_gain=2.5, _explicit_cli_dests={'denoise_gain'})
        final = [SimpleNamespace(header={'EGAIN': 1.2}, metrics={})]
        a._apply_quality_settings(sig, args, weights={}, final=final)
        assert args.denoise_gain == 2.5


class TestGaiaDistortionAutoRule:
    """--auto folding: gaia_distortion_correction=True unconditionally
    (the function itself self-gates on WCS presence and fails soft, so no
    frame-count/star-count threshold is needed at the auto_settings level)."""

    def test_enables_by_default(self):
        sig = _sig()
        args = _args()
        a._apply_quality_settings(sig, args, weights={})
        assert args.gaia_distortion_correction is True

    def test_enables_regardless_of_frame_count(self):
        sig = _sig(n_frames=3, star_count=2)
        args = _args()
        a._apply_quality_settings(sig, args, weights={})
        assert args.gaia_distortion_correction is True

    def test_explicit_flag_not_overridden(self):
        sig = _sig()
        args = _args(gaia_distortion_correction=False,
                     _explicit_cli_dests={'gaia_distortion_correction'})
        a._apply_quality_settings(sig, args, weights={})
        assert args.gaia_distortion_correction is False

    def test_already_true_stays_true(self):
        sig = _sig()
        args = _args(gaia_distortion_correction=True)
        a._apply_quality_settings(sig, args, weights={})
        assert args.gaia_distortion_correction is True


def _final_with_bortle(bortle_vals):
    return [SimpleNamespace(header={}, metrics={'bortle_estimate': b}) for b in bortle_vals]


class TestBortleAwareBackgroundAutoRule:
    """--auto folding: estimate_bortle (src/quality.py) now actually drives
    a processing decision (background-extraction aggressiveness) instead of
    only feeding the end-of-run summary print."""

    def test_low_bortle_no_change(self):
        sig = _sig()
        args = _args()
        final = _final_with_bortle([3, 3, 4])
        a._apply_quality_settings(sig, args, weights={}, final=final)
        assert args.pre_gradient_removal is False
        assert args.dbe_patch_size == 64
        assert args.bg_clip_sigma == 3.0
        assert args.local_normalize is False

    def test_bortle_7_tightens_settings(self):
        sig = _sig()
        args = _args()
        final = _final_with_bortle([7, 7, 8])
        a._apply_quality_settings(sig, args, weights={}, final=final)
        assert args.pre_gradient_removal is True
        assert args.dbe_patch_size <= 48
        assert args.bg_clip_sigma <= 2.5
        assert args.local_normalize is True

    def test_bortle_9_tightens_further(self):
        sig = _sig()
        args = _args()
        final = _final_with_bortle([9, 9, 9])
        a._apply_quality_settings(sig, args, weights={}, final=final)
        assert args.pre_gradient_removal is True
        assert args.dbe_patch_size <= 32

    def test_acts_as_ceiling_not_override(self):
        """If the target-type blend already picked a tighter dbe_patch_size
        than the bortle rule would, bortle must not loosen it back up."""
        sig = _sig()
        args = _args(dbe_patch_size=20)
        final = _final_with_bortle([7, 7, 7])
        a._apply_quality_settings(sig, args, weights={}, final=final)
        assert args.dbe_patch_size == 20

    def test_no_bortle_data_no_change(self):
        sig = _sig()
        args = _args()
        final = [SimpleNamespace(header={}, metrics={})]
        a._apply_quality_settings(sig, args, weights={}, final=final)
        assert args.pre_gradient_removal is False
        assert args.dbe_patch_size == 64

    def test_explicit_pre_gradient_removal_not_overridden(self):
        sig = _sig()
        args = _args(pre_gradient_removal=False,
                     _explicit_cli_dests={'pre_gradient_removal'})
        final = _final_with_bortle([9, 9, 9])
        a._apply_quality_settings(sig, args, weights={}, final=final)
        assert args.pre_gradient_removal is False

    def test_explicit_local_normalize_not_overridden(self):
        sig = _sig()
        args = _args(local_normalize=False,
                     _explicit_cli_dests={'local_normalize'})
        final = _final_with_bortle([9, 9, 9])
        a._apply_quality_settings(sig, args, weights={}, final=final)
        assert args.local_normalize is False

    def test_local_normalize_already_true_stays_true(self):
        sig = _sig()
        args = _args(local_normalize=True)
        final = _final_with_bortle([7, 7, 7])
        a._apply_quality_settings(sig, args, weights={}, final=final)
        assert args.local_normalize is True

    def test_low_bortle_leaves_local_normalize_off(self):
        sig = _sig()
        args = _args()
        final = _final_with_bortle([2, 3, 3])
        a._apply_quality_settings(sig, args, weights={}, final=final)
        assert args.local_normalize is False
