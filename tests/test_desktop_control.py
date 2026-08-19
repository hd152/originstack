"""Tests for src/desktop_control.py -- the desktop-app control layer
(form schema introspection, form-to-argv translation, RunManager)."""
import unittest

from src.cli import parse_args
from src.desktop_control import (get_form_schema, build_argv_from_form,
                                 RunManager, get_run_manager)


class TestFormSchema(unittest.TestCase):
    def test_groups_and_fields_present(self):
        schema = get_form_schema()
        self.assertIn('Core', schema)
        dests = {f['dest'] for fields in schema.values() for f in fields}
        self.assertIn('directory', dests)
        self.assertIn('stack_method', dests)
        self.assertIn('trail_reject', dests)

    def test_empty_advanced_group_omitted(self):
        # g_adv has zero add_argument calls today -- schema should not
        # include an empty group.
        schema = get_form_schema()
        for fields in schema.values():
            self.assertTrue(fields)

    def test_bool_and_select_field_kinds(self):
        schema = get_form_schema()
        by_dest = {f['dest']: f for fields in schema.values() for f in fields}
        self.assertEqual(by_dest['trail_reject']['kind'], 'bool_true')
        self.assertEqual(by_dest['auto']['kind'], 'bool_false')
        self.assertEqual(by_dest['stack_method']['kind'], 'select')
        self.assertIn('sigma_clip', by_dest['stack_method']['choices'])

    def test_unsupported_dests_excluded(self):
        schema = get_form_schema()
        dests = {f['dest'] for fields in schema.values() for f in fields}
        self.assertNotIn('live', dests)
        self.assertNotIn('stream', dests)


class TestBuildArgvFromForm(unittest.TestCase):
    def test_empty_form_yields_empty_argv(self):
        self.assertEqual(build_argv_from_form({}), [])

    def test_none_values_skipped(self):
        self.assertEqual(build_argv_from_form({'trail_reject': None}), [])

    def test_store_true_bool(self):
        argv = build_argv_from_form({'trail_reject': True})
        self.assertEqual(argv, ['--trail-reject'])
        argv = build_argv_from_form({'trail_reject': False})
        self.assertEqual(argv, [])

    def test_store_false_bool(self):
        argv = build_argv_from_form({'auto': False})
        self.assertEqual(argv, ['--no-auto'])
        argv = build_argv_from_form({'auto': True})
        self.assertEqual(argv, [])

    def test_select_and_number(self):
        argv = build_argv_from_form({'stack_method': 'sigma_clip', 'rejection_sigma': 3.5})
        self.assertIn('--stack-method', argv)
        self.assertIn('sigma_clip', argv)
        self.assertIn('--rejection-sigma', argv)
        self.assertIn('3.5', argv)

    def test_nargs_plus_list_from_comma_string(self):
        argv = build_argv_from_form({'merge': 'a.fits, b.fits'})
        self.assertEqual(argv, ['--merge', 'a.fits', 'b.fits'])

    def test_append_action_repeats_flag_per_token(self):
        """--skip-step is action='append' (one value per occurrence), a
        distinct code path from nargs='+' (--merge, one flag then all
        tokens) -- must produce a separate --skip-step per token, not a
        single one followed by bare values (which argparse would reject)."""
        argv = build_argv_from_form({'skip_step': 'sky_pedestal, sky_neutralize'})
        self.assertEqual(argv, ['--skip-step', 'sky_pedestal',
                                '--skip-step', 'sky_neutralize'])
        args = parse_args(['-d', 'x', '-o', 'y.fits'] + argv)
        self.assertEqual(args.skip_step, ['sky_pedestal', 'sky_neutralize'])

    def test_unsupported_dest_ignored(self):
        self.assertEqual(build_argv_from_form({'live': True}), [])

    def test_unknown_dest_ignored(self):
        self.assertEqual(build_argv_from_form({'not_a_real_flag': 'x'}), [])

    def test_round_trips_through_parse_args_with_correct_explicit_dests(self):
        argv = build_argv_from_form({
            'directory': 'foo', 'output': 'bar.fits',
            'trail_reject': True, 'auto': False, 'stack_method': 'sigma_clip',
        })
        args = parse_args(argv)
        self.assertEqual(args.directory, 'foo')
        self.assertEqual(args.output, 'bar.fits')
        self.assertTrue(args.trail_reject)
        self.assertFalse(args.auto)
        self.assertEqual(args.stack_method, 'sigma_clip')
        self.assertEqual(
            args._explicit_cli_dests,
            {'directory', 'output', 'trail_reject', 'auto', 'stack_method'})

    def test_untouched_fields_stay_at_default_and_non_explicit(self):
        argv = build_argv_from_form({'directory': 'foo', 'output': 'bar.fits'})
        args = parse_args(argv)
        self.assertEqual(args.stack_method, 'auto')  # argparse default
        self.assertNotIn('stack_method', args._explicit_cli_dests)


class TestRunManager(unittest.TestCase):
    def test_singleton(self):
        self.assertIs(get_run_manager(), get_run_manager())

    def test_is_running_reflects_status(self):
        rm = RunManager()
        self.assertFalse(rm.is_running())
        rm.status = 'running'
        self.assertTrue(rm.is_running())
        for s in ('idle', 'ok', 'error'):
            rm.status = s
            self.assertFalse(rm.is_running())

    def test_rejects_concurrent_start(self):
        rm = RunManager()
        rm.status = 'running'
        result = rm.start({'directory': 'foo', 'output': 'bar.fits'})
        self.assertFalse(result['ok'])
        self.assertIn('already in progress', result['error'])

    def test_start_runs_in_background_thread_and_transitions_to_ok(self):
        """The happy path RunManager.start() -> idle -> running -> ok, as
        actually exercised by a real (mocked-pipeline) background thread --
        distinct from test_rejects_concurrent_start (which never spawns a
        thread) and test_run_catches_systemexit_... (which calls _run()
        directly, synchronously, bypassing start() entirely)."""
        from unittest.mock import patch
        rm = RunManager()
        self.assertEqual(rm.status, 'idle')
        # Keep the patch active for the join too -- it must still be in
        # effect when the background thread reaches its own
        # `from src.cli import process_directory` (a fresh call-time import,
        # per RunManager._run's design), which can happen anytime after
        # start() returns, not necessarily before this `with` block would
        # otherwise exit.
        with patch('src.cli.process_directory') as mock_pd:
            result = rm.start({'directory': 'foo', 'output': 'bar.fits'})
            self.assertTrue(result['ok'])
            self.assertIsNotNone(rm.thread)
            rm.thread.join(timeout=5)
            self.assertFalse(rm.thread.is_alive())
            mock_pd.assert_called_once_with('foo', 'bar.fits', mock_pd.call_args[0][2])
        self.assertEqual(rm.status, 'ok')

    def test_start_transitions_to_error_on_pipeline_exception(self):
        from unittest.mock import patch
        rm = RunManager()
        with patch('src.cli.process_directory', side_effect=RuntimeError('boom')):
            result = rm.start({'directory': 'foo', 'output': 'bar.fits'})
            self.assertTrue(result['ok'])
            rm.thread.join(timeout=5)
        self.assertEqual(rm.status, 'error')

    def test_run_catches_systemexit_from_missing_required_directory(self):
        # 'directory' is required by the parser -- parse_args([]) raises
        # SystemExit via argparse's own error() path. Call _run directly
        # (synchronous, no thread) so the SystemExit-vs-Exception handling
        # in RunManager._run is tested deterministically rather than racing
        # a background thread.
        rm = RunManager()
        rm.status = 'running'
        rm._run([])
        self.assertEqual(rm.status, 'error')


if __name__ == '__main__':
    unittest.main()
