"""--auto must advise each target on its own merits.

``apply_auto_settings`` mutates ``args`` in place, and ``process_directory``
loops every target against one shared ``args`` object. Most settings survive
that because the advisor recomputes a value for each target and overwrites the
previous one. Accumulating settings do not.

``skip_step`` is the case that bites: it is *appended* to, and the append is
guarded by "not already present", so once a nebula target adds 'sky_residual'
every later target sees it there and leaves it alone. A globular cluster
stacked from the same directory then silently skips its sky-residual
correction because a different session, of a different object, wanted that.

This matters more than it used to: multi-night directories now default to
per-session stacking, so far more runs walk this loop.
"""

import argparse
import copy
import unittest

from src import auto_settings as a


def _fresh_args(**overrides):
    base = dict(
        _explicit_cli_dests=set(), stack_method='auto', deconvolve=False,
        auto_denoise_strength=True, debayer_method='malvar', denoise_mmt=False,
        denoise_acdnr=False, denoise=False, denoise_curvelet=False,
        denoise_bm3d=False, deconvolve_tv=False, patch_registration=False,
        consensus_ref=False, preview_black_sigma=0.0, variance_stabilize=False,
        drizzle_scale=1.0, drizzle_kernel='lanczos3', hdr_combine=None,
        hdr_blend_mode='threshold', color_calibrate=False,
        color_calibrate_method='colorindex', star_reduce=True,
        local_contrast=True, skip_step=[], deconvolve_blind_psf=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _advise(args, anchor):
    """Run the real advisor for one target type against ``args``."""
    signals = dict(a._TYPE_ANCHORS[anchor])
    weights = a._blend_weights(signals)
    a._apply_dynamic_settings(signals, weights, args)
    return args


def _restore(args, baseline):
    """What process_directory does between targets."""
    for k, v in baseline.items():
        setattr(args, k, copy.copy(v) if isinstance(v, (list, dict, set)) else v)


def _snapshot(args):
    return {k: (copy.copy(v) if isinstance(v, (list, dict, set)) else v)
            for k, v in vars(args).items()}


class TestSkipStepDoesNotLeakBetweenTargets(unittest.TestCase):

    def test_nebula_target_adds_the_sky_residual_skip(self):
        """Establishes the precondition the leak depends on."""
        args = _advise(_fresh_args(), 'emission_nebula')
        self.assertIn('sky_residual', args.skip_step)

    def test_a_cluster_on_its_own_does_not_skip_sky_residual(self):
        args = _advise(_fresh_args(), 'globular_cluster')
        self.assertNotIn('sky_residual', args.skip_step)

    def test_shared_args_without_a_reset_leak_the_skip(self):
        """The bug, reproduced: this is what the loop used to do."""
        shared = _fresh_args()
        _advise(shared, 'emission_nebula')
        _advise(shared, 'globular_cluster')

        self.assertIn('sky_residual', shared.skip_step,
                      "precondition: without a reset the skip persists")

    def test_restoring_the_baseline_isolates_each_target(self):
        shared = _fresh_args()
        baseline = _snapshot(shared)

        _restore(shared, baseline)
        _advise(shared, 'emission_nebula')
        nebula_skip = list(shared.skip_step)

        _restore(shared, baseline)
        _advise(shared, 'globular_cluster')

        self.assertIn('sky_residual', nebula_skip)
        self.assertNotIn('sky_residual', shared.skip_step,
                         "the cluster target inherited the nebula's skip")

    def test_the_baseline_list_is_copied_not_aliased(self):
        """Restoring must not hand back the list the last target appended to.

        A shallow restore that aliases the same list object would leave the
        baseline itself polluted after the first target, so every later
        restore would replay the leak.
        """
        shared = _fresh_args()
        baseline = _snapshot(shared)

        _restore(shared, baseline)
        _advise(shared, 'emission_nebula')

        self.assertEqual(baseline['skip_step'], [],
                         "the advisor mutated the baseline through an alias")

    def test_an_explicit_user_skip_survives_every_target(self):
        """A --skip-step the user typed is not the advisor's to discard."""
        shared = _fresh_args(skip_step=['sky_pedestal'],
                             _explicit_cli_dests={'skip_step'})
        baseline = _snapshot(shared)

        for anchor in ('emission_nebula', 'globular_cluster', 'galaxy'):
            _restore(shared, baseline)
            _advise(shared, anchor)
            with self.subTest(anchor=anchor):
                self.assertIn('sky_pedestal', shared.skip_step)
                self.assertNotIn('sky_residual', shared.skip_step,
                                 "explicit --skip-step opts out of auto's append")


if __name__ == '__main__':
    unittest.main()
