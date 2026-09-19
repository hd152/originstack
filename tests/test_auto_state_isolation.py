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

Every test here exercises the production ``_snapshot_args``/``_restore_args``
from ``src.cli``, and one drives ``process_directory`` itself. An earlier
version tested local copies of those helpers defined in this file, which made
it tautological: removing the copy-not-alias rule from ``cli.py`` left every
test green, because the rule under test lived in the test.
"""

import argparse
import os
import tempfile
import unittest
from unittest import mock

from src import auto_settings as a
from src import cli
from src.cli import _restore_args, _snapshot_args


def _fresh_args(**overrides):
    base = dict(
        _explicit_cli_dests=set(), stack_method='auto', deconvolve=False,
        debayer_method='malvar',
        denoise_acdnr=False, denoise_curvelet=False,
        deconvolve_tv=False, patch_registration=False,
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
        baseline = _snapshot_args(shared)

        _restore_args(shared, baseline)
        _advise(shared, 'emission_nebula')
        nebula_skip = list(shared.skip_step)

        _restore_args(shared, baseline)
        _advise(shared, 'globular_cluster')

        self.assertIn('sky_residual', nebula_skip)
        self.assertNotIn('sky_residual', shared.skip_step,
                         "the cluster target inherited the nebula's skip")

    def test_the_snapshot_does_not_alias_the_live_list(self):
        """Half one of the copy rule: snapshot must copy.

        If the snapshot held the same list object as ``args``, the first
        target's appends would pollute the baseline itself, and every later
        restore would faithfully replay the leak.
        """
        shared = _fresh_args()
        baseline = _snapshot_args(shared)
        shared.skip_step.append('sky_residual')
        self.assertEqual(baseline['skip_step'], [],
                         "appending to args mutated the baseline through an alias")

    def test_the_restore_does_not_alias_the_baseline_list(self):
        """Half two of the copy rule: restore must copy too.

        Restoring by reference hands the next target the baseline's own list,
        so its append lands in the baseline and leaks into every target after.
        """
        shared = _fresh_args()
        baseline = _snapshot_args(shared)

        _restore_args(shared, baseline)
        _advise(shared, 'emission_nebula')

        self.assertEqual(baseline['skip_step'], [],
                         "the advisor mutated the baseline through an alias")

    def test_an_explicit_user_skip_survives_every_target(self):
        """A --skip-step the user typed is not the advisor's to discard."""
        shared = _fresh_args(skip_step=['sky_pedestal'],
                             _explicit_cli_dests={'skip_step'})
        baseline = _snapshot_args(shared)

        for anchor in ('emission_nebula', 'globular_cluster', 'galaxy'):
            _restore_args(shared, baseline)
            _advise(shared, anchor)
            with self.subTest(anchor=anchor):
                self.assertIn('sky_pedestal', shared.skip_step)
                self.assertNotIn('sky_residual', shared.skip_step,
                                 "explicit --skip-step opts out of auto's append")


class TestAttributesCreatedByATargetDoNotLeak(unittest.TestCase):
    """The same bug one layer further out: attributes that did not exist in
    the baseline at all, created conditionally during a target's run."""

    def test_an_attribute_created_during_a_target_is_removed(self):
        shared = _fresh_args()
        baseline = _snapshot_args(shared)

        _restore_args(shared, baseline)
        # Set by cli._build_masters only when that target had bias + flat
        # pairs and the photon-transfer fit succeeded.
        shared._ptc_gain_e_per_adu = 1.7

        _restore_args(shared, baseline)
        self.assertFalse(
            hasattr(shared, '_ptc_gain_e_per_adu'),
            "a calibration-less target would reuse the previous target's "
            "measured gain for --photometry's Poisson term")

    def test_baseline_attributes_are_kept(self):
        shared = _fresh_args()
        baseline = _snapshot_args(shared)
        shared._scratch = True
        _restore_args(shared, baseline)
        for key in baseline:
            with self.subTest(key=key):
                self.assertTrue(hasattr(shared, key))


class TestProcessDirectoryResetsBetweenTargets(unittest.TestCase):
    """Drives the real ``process_directory`` loop, not a model of it.

    The heavy stages are stubbed and ``--dry-run`` makes each target stop
    before stacking, so this is fast -- but the snapshot, the restore and the
    loop that calls them are all production code. ``_build_masters`` is the
    patch point because it is exactly where the real ``_ptc_gain_e_per_adu``
    leak originates.
    """

    def _run(self, n_targets=2):
        seen = []

        def fake_build_masters(frames, stats, args):
            # Record what this target starts with, then leave behind what a
            # real target does: an accumulated skip and a created attribute.
            seen.append({
                'skip_step': list(args.skip_step),
                'has_ptc_gain': hasattr(args, '_ptc_gain_e_per_adu'),
            })
            args.skip_step.append('sky_residual')
            args._ptc_gain_e_per_adu = 1.7
            return {}

        fake_light = mock.Mock(header={'NAXIS1': 8, 'NAXIS2': 8})
        with tempfile.TemporaryDirectory() as root:
            for i in range(n_targets):
                os.mkdir(os.path.join(root, f"session{i}"))
            args = argparse.Namespace(
                skip_step=[], hierarchical=True, mosaic=False,
                combine_sessions=False, dry_run=True, health_check=False,
                preset=None, verbose=False, stack_method='auto',
                _explicit_cli_dests=set())

            with mock.patch.object(cli, 'discover_frames', return_value={
                        'light': [fake_light], 'dark': [], 'flat': [], 'bias': []}), \
                 mock.patch.object(cli, '_load_calibration_dir', return_value={
                        'dark': [], 'flat': [], 'bias': []}), \
                 mock.patch.object(cli, 'group_lights_by_filter',
                                   side_effect=lambda lights: {'L': lights}), \
                 mock.patch.object(cli, '_build_masters', side_effect=fake_build_masters), \
                 mock.patch.object(cli, 'stack_target',
                                   side_effect=AssertionError("dry run must not stack")):
                cli.process_directory(root, os.path.join(root, 'out.fits'), args)
        return seen

    def test_every_target_starts_from_the_baseline(self):
        seen = self._run(n_targets=3)
        self.assertEqual(len(seen), 3, "every target should have been visited")
        for idx, state in enumerate(seen, 1):
            with self.subTest(target=idx):
                self.assertEqual(state['skip_step'], [],
                                 "a previous target's skip_step append leaked in")
                self.assertFalse(state['has_ptc_gain'],
                                 "a previous target's measured gain leaked in")


if __name__ == '__main__':
    unittest.main()
