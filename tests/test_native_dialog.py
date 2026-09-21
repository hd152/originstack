"""Tests for src/native_dialog.py -- the ctypes MessageBoxW wrapper that
replaced tkinter in desktop_app.py on Windows (saves ~7-8MB of bundled Tcl/Tk runtime
in the packaged build). Off Windows it uses tkinter's own boxes; those tests replace
``_tk_box`` so a real dialog can never open during the suite."""
from __future__ import annotations

import sys
import unittest
from unittest import mock

import src.native_dialog as nd
from src.native_dialog import IDYES, ask_yes_no, show_error


class TestShowError(unittest.TestCase):
    def test_non_windows_uses_the_tk_box(self):
        with mock.patch.object(sys, 'platform', 'linux'),              mock.patch.object(nd, '_tk_box', return_value=True) as box:
            self.assertTrue(show_error("title", "message"))
        box.assert_called_once_with("error", "title", "message")

    def test_non_windows_without_a_display_reports_not_shown(self):
        with mock.patch.object(sys, 'platform', 'linux'),              mock.patch.object(nd, '_tk_box', return_value=None):
            self.assertFalse(show_error("title", "message"))

    def test_calls_message_box_and_returns_true(self):
        fake_user32 = mock.Mock()
        fake_windll = mock.Mock()
        fake_windll.user32 = fake_user32
        with mock.patch.object(sys, 'platform', 'win32'), \
             mock.patch('ctypes.windll', create=True, new=fake_windll):
            self.assertTrue(show_error("OriginStack", "Something failed"))
        fake_user32.MessageBoxW.assert_called_once()
        args = fake_user32.MessageBoxW.call_args[0]
        self.assertEqual(args[1], "Something failed")
        self.assertEqual(args[2], "OriginStack")

    def test_swallows_any_exception(self):
        fake_user32 = mock.Mock()
        fake_user32.MessageBoxW.side_effect = OSError("no user32 here")
        fake_windll = mock.Mock()
        fake_windll.user32 = fake_user32
        with mock.patch.object(sys, 'platform', 'win32'), \
             mock.patch('ctypes.windll', create=True, new=fake_windll):
            self.assertFalse(show_error("title", "message"))


class TestAskYesNo(unittest.TestCase):
    def test_non_windows_returns_the_users_answer(self):
        with mock.patch.object(sys, 'platform', 'linux'):
            with mock.patch.object(nd, '_tk_box', return_value=False):
                self.assertFalse(ask_yes_no("t", "m", default=True))
            with mock.patch.object(nd, '_tk_box', return_value=True):
                self.assertTrue(ask_yes_no("t", "m", default=False))

    def test_non_windows_falls_back_to_default_without_a_display(self):
        with mock.patch.object(sys, 'platform', 'linux'),              mock.patch.object(nd, '_tk_box', return_value=None):
            self.assertTrue(ask_yes_no("t", "m", default=True))
            self.assertFalse(ask_yes_no("t", "m", default=False))

    def test_returns_true_when_user_clicks_yes(self):
        fake_user32 = mock.Mock()
        fake_user32.MessageBoxW.return_value = IDYES
        fake_windll = mock.Mock()
        fake_windll.user32 = fake_user32
        with mock.patch.object(sys, 'platform', 'win32'), \
             mock.patch('ctypes.windll', create=True, new=fake_windll):
            self.assertTrue(ask_yes_no("t", "m"))

    def test_returns_false_when_user_clicks_no(self):
        fake_user32 = mock.Mock()
        fake_user32.MessageBoxW.return_value = 7  # IDNO
        fake_windll = mock.Mock()
        fake_windll.user32 = fake_user32
        with mock.patch.object(sys, 'platform', 'win32'), \
             mock.patch('ctypes.windll', create=True, new=fake_windll):
            self.assertFalse(ask_yes_no("t", "m"))

    def test_returns_default_when_dialog_fails(self):
        fake_user32 = mock.Mock()
        fake_user32.MessageBoxW.side_effect = OSError("boom")
        fake_windll = mock.Mock()
        fake_windll.user32 = fake_user32
        with mock.patch.object(sys, 'platform', 'win32'), \
             mock.patch('ctypes.windll', create=True, new=fake_windll):
            self.assertTrue(ask_yes_no("t", "m", default=True))
            self.assertFalse(ask_yes_no("t", "m", default=False))


if __name__ == '__main__':
    unittest.main()
