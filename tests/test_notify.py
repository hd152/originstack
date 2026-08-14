"""Tests for src/notify.py -- the ctypes Windows balloon-tip notification."""
from __future__ import annotations

import sys
import unittest
from unittest import mock

from src.notify import notify_windows


class TestNotifyWindows(unittest.TestCase):
    def test_noop_on_non_windows(self):
        with mock.patch.object(sys, 'platform', 'linux'):
            self.assertFalse(notify_windows("title", "message"))

    def test_swallows_any_exception(self):
        fake_shell32 = mock.Mock()
        fake_shell32.Shell_NotifyIconW.side_effect = OSError("no shell32 here")
        fake_windll = mock.Mock()
        fake_windll.shell32 = fake_shell32
        with mock.patch.object(sys, 'platform', 'win32'), \
             mock.patch('ctypes.windll', create=True, new=fake_windll):
            self.assertFalse(notify_windows("title", "message"))

    def test_returns_true_on_success(self):
        fake_shell32 = mock.Mock()
        fake_shell32.Shell_NotifyIconW.return_value = 1
        fake_windll = mock.Mock()
        fake_windll.shell32 = fake_shell32
        with mock.patch.object(sys, 'platform', 'win32'), \
             mock.patch('ctypes.windll', create=True, new=fake_windll):
            self.assertTrue(notify_windows("OriginStack", "Run complete."))
        self.assertEqual(fake_shell32.Shell_NotifyIconW.call_count, 2)  # NIM_ADD + NIM_DELETE

    def test_returns_false_when_add_fails(self):
        fake_shell32 = mock.Mock()
        fake_shell32.Shell_NotifyIconW.return_value = 0
        fake_windll = mock.Mock()
        fake_windll.shell32 = fake_shell32
        with mock.patch.object(sys, 'platform', 'win32'), \
             mock.patch('ctypes.windll', create=True, new=fake_windll):
            self.assertFalse(notify_windows("OriginStack", "Run complete."))

    def test_truncates_long_message_and_title(self):
        fake_shell32 = mock.Mock()
        fake_shell32.Shell_NotifyIconW.return_value = 1
        fake_windll = mock.Mock()
        fake_windll.shell32 = fake_shell32
        long_title = "T" * 200
        long_message = "M" * 500
        with mock.patch.object(sys, 'platform', 'win32'), \
             mock.patch('ctypes.windll', create=True, new=fake_windll):
            # Must not raise even with over-length strings (struct field caps).
            self.assertTrue(notify_windows(long_title, long_message))


if __name__ == '__main__':
    unittest.main()
