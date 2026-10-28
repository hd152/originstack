"""Tests for src/net_query.py -- the direct-HTTP replacements for every
astroquery service this codebase used (Gaia/VizieR/SIMBAD TAP, astrometry.net,
JPL Horizons).

These tests mock the HTTP layer (``urllib.request.urlopen``) and validate
the *parsing* logic against realistic response payloads -- the part fully
under this codebase's control. They do NOT validate against the live
services (no network access in this environment): the exact endpoint URLs
and response schemas are based on each service's public documentation, not
a live-verified integration test. Any query function returning None on
malformed/unexpected input is itself part of the contract (matches the
astroquery-backed code's existing best-effort, non-fatal behaviour), so a
schema drift degrades gracefully rather than crashing the pipeline.
"""
from __future__ import annotations

import io
import json
import urllib.parse
from unittest import mock

import numpy as np
import pytest

from src import net_query


def _fake_response(data: bytes):
    """A context-manager-compatible fake matching urlopen()'s return value."""
    m = mock.MagicMock()
    m.read.return_value = data
    m.__enter__.return_value = m
    m.__exit__.return_value = False
    return m


class TestTapQuery:
    def test_parses_standard_tap_json(self):
        payload = {
            "metadata": [{"name": "ra"}, {"name": "dec"}, {"name": "phot_g_mean_mag"}],
            "data": [[10.5, -5.25, 12.3], [11.0, -5.30, 13.1]],
        }
        with mock.patch("urllib.request.urlopen",
                        return_value=_fake_response(json.dumps(payload).encode())):
            table = net_query.tap_query("https://example.invalid/tap/sync", "SELECT 1")
        assert table is not None
        assert len(table) == 2
        assert list(table["ra"]) == [10.5, 11.0]
        assert list(table["dec"]) == [-5.25, -5.30]

    def test_null_values_become_nan(self):
        payload = {
            "metadata": [{"name": "ra"}, {"name": "phot_bp_mean_mag"}],
            "data": [[10.5, None], [11.0, 14.2]],
        }
        with mock.patch("urllib.request.urlopen",
                        return_value=_fake_response(json.dumps(payload).encode())):
            table = net_query.tap_query("https://example.invalid/tap/sync", "SELECT 1")
        assert table is not None
        assert np.isnan(table["phot_bp_mean_mag"][0])
        assert table["phot_bp_mean_mag"][1] == pytest.approx(14.2)

    def test_string_column_preserved(self):
        payload = {
            "metadata": [{"name": "main_id"}, {"name": "otype"}],
            "data": [["M 42", "HII"]],
        }
        with mock.patch("urllib.request.urlopen",
                        return_value=_fake_response(json.dumps(payload).encode())):
            table = net_query.tap_query("https://example.invalid/tap/sync", "SELECT 1")
        assert table is not None
        assert str(table["main_id"][0]) == "M 42"
        assert str(table["otype"][0]) == "HII"

    def test_malformed_response_returns_none(self):
        with mock.patch("urllib.request.urlopen",
                        return_value=_fake_response(b"not json")):
            table = net_query.tap_query("https://example.invalid/tap/sync", "SELECT 1")
        assert table is None

    def test_network_error_returns_none(self):
        with mock.patch("urllib.request.urlopen", side_effect=OSError("no route")):
            table = net_query.tap_query("https://example.invalid/tap/sync", "SELECT 1")
        assert table is None


class TestGaiaVizierSimbadQueries:
    def test_gaia_cone_search_builds_adql_and_parses(self):
        payload = {
            "metadata": [{"name": "ra"}, {"name": "dec"}, {"name": "phot_g_mean_mag"}],
            "data": [[10.0, 20.0, 15.0]],
        }
        captured = {}

        def _fake_urlopen(req, timeout=None, context=None):
            captured["url"] = req.full_url
            captured["body"] = req.data
            return _fake_response(json.dumps(payload).encode())

        with mock.patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            table = net_query.gaia_cone_search(
                10.0, 20.0, 0.5,
                columns=["ra", "dec", "phot_g_mean_mag"],
                require_not_null=["phot_g_mean_mag"])
        assert table is not None
        assert len(table) == 1
        adql = urllib.parse.parse_qs(captured["body"].decode())["QUERY"][0]
        assert "gaiadr3.gaia_source" in adql
        assert "CONTAINS" in adql
        assert "IS NOT NULL" in adql
        # TOP n without ORDER BY returns an arbitrary subset of a crowded cone, a
        # different one per call (72-78 matches for one stack on four runs)
        assert adql.rstrip().endswith("ORDER BY phot_g_mean_mag ASC")
        assert adql.index("WHERE") < adql.index("ORDER BY")

    def test_cone_search_order_by_is_validated_and_optional(self):
        captured = []

        def _fake_urlopen(req, timeout=None, context=None):
            captured.append(urllib.parse.parse_qs(req.data.decode())["QUERY"][0])
            return _fake_response(json.dumps({"metadata": [{"name": "ra"}], "data": [[1.0]]}).encode())

        with mock.patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            net_query.gaia_cone_search(1.0, 2.0, 0.1, columns=["ra"], order_by=None)
            net_query.vizier_cone_search(1.0, 2.0, 0.1, "II/246/out", ["RAJ2000"], order_by="Jmag")
        assert "ORDER BY" not in captured[0]
        assert captured[1].rstrip().endswith('ORDER BY "Jmag" ASC')
        with pytest.raises(ValueError):
            net_query.gaia_cone_search(1.0, 2.0, 0.1, columns=["ra"], order_by="g; DROP")

    def test_simbad_name_lookup_escapes_quotes(self):
        payload = {"metadata": [{"name": "main_id"}, {"name": "otype"}],
                   "data": [["NGC 1976", "HII"]]}
        captured = {}

        def _fake_urlopen(req, timeout=None, context=None):
            captured["body"] = req.data
            return _fake_response(json.dumps(payload).encode())

        with mock.patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            table = net_query.simbad_name_lookup("O'Brien's Nebula")
        assert table is not None
        adql = urllib.parse.parse_qs(captured["body"].decode())["QUERY"][0]
        # single quote must be doubled for ADQL string-literal escaping
        assert "O''Brien''s Nebula" in adql


class TestAstrometryNetFlow:
    def test_login_success(self):
        payload = {"status": "success", "session": "abc123"}
        with mock.patch("urllib.request.urlopen",
                        return_value=_fake_response(json.dumps(payload).encode())):
            session = net_query.astrometry_net_login("fake-key")
        assert session == "abc123"

    def test_login_failure_returns_none(self):
        payload = {"status": "error", "errormessage": "bad key"}
        with mock.patch("urllib.request.urlopen",
                        return_value=_fake_response(json.dumps(payload).encode())):
            session = net_query.astrometry_net_login("bad-key")
        assert session is None

    def test_poll_submission_finds_job(self):
        payload = {"jobs": [42]}
        with mock.patch("urllib.request.urlopen",
                        return_value=_fake_response(json.dumps(payload).encode())):
            job_id = net_query.astrometry_net_poll_submission(
                "sub1", deadline=__import__("time").time() + 10, poll_interval=0.01)
        assert job_id == "42"

    def test_poll_submission_no_job_yet_times_out(self):
        payload = {"jobs": [None]}
        with mock.patch("urllib.request.urlopen",
                        return_value=_fake_response(json.dumps(payload).encode())):
            job_id = net_query.astrometry_net_poll_submission(
                "sub1", deadline=__import__("time").time() + 0.02, poll_interval=0.01)
        assert job_id is None

    def test_poll_job_success(self):
        payload = {"status": "success"}
        with mock.patch("urllib.request.urlopen",
                        return_value=_fake_response(json.dumps(payload).encode())):
            ok = net_query.astrometry_net_poll_job(
                "job1", deadline=__import__("time").time() + 10, poll_interval=0.01)
        assert ok is True

    def test_poll_job_failure(self):
        payload = {"status": "failure"}
        with mock.patch("urllib.request.urlopen",
                        return_value=_fake_response(json.dumps(payload).encode())):
            ok = net_query.astrometry_net_poll_job(
                "job1", deadline=__import__("time").time() + 10, poll_interval=0.01)
        assert ok is False


class TestHorizonsEphemeris:
    def _payload(self, ra=83.822083, dec=-5.391111):
        result_text = (
            "*******************************************************************\n"
            "$$SOE\n"
            f" 2024-Jan-01 00:00,  ,   {ra}, {dec},\n"
            "$$EOE\n"
            "*******************************************************************\n"
        )
        return {"result": result_text}

    def test_parses_ra_dec(self):
        with mock.patch("urllib.request.urlopen",
                        return_value=_fake_response(json.dumps(self._payload()).encode())):
            radec = net_query.horizons_ephemeris("C/2023 A3", "2024-01-01T00:00:00")
        assert radec is not None
        ra, dec = radec
        assert ra == pytest.approx(83.822083)
        assert dec == pytest.approx(-5.391111)

    def test_missing_markers_returns_none(self):
        with mock.patch("urllib.request.urlopen",
                        return_value=_fake_response(json.dumps({"result": "no data"}).encode())):
            radec = net_query.horizons_ephemeris("C/2023 A3", "2024-01-01T00:00:00")
        assert radec is None

    def test_bad_time_string_returns_none(self):
        radec = net_query.horizons_ephemeris("C/2023 A3", "not-a-time")
        assert radec is None

    def test_topocentric_location_sets_site_coord(self):
        captured = {}

        def _fake_urlopen(url, timeout=None, context=None):
            captured["url"] = url if isinstance(url, str) else url.full_url
            return _fake_response(json.dumps(self._payload()).encode())

        with mock.patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            radec = net_query.horizons_ephemeris(
                "C/2023 A3", "2024-01-01T00:00:00",
                location={"lon": -2.5, "lat": 51.4, "elevation": 0.05})
        assert radec is not None
        assert "SITE_COORD" in captured["url"]
        assert "COORD_TYPE" in captured["url"]


class TestCheckForUpdate:
    """src.net_query.check_for_update -- the CLI/desktop-app self-update check
    (src.utils.should_check_for_update gates whether it's called at all)."""

    def _mock_release(self, tag):
        payload = {"tag_name": tag, "html_url": f"https://github.com/hd152/originstack/releases/tag/{tag}"}
        return _fake_response(json.dumps(payload).encode())

    def teardown_method(self):
        net_query.set_offline(False)  # this module's offline flag is process-global

    def test_newer_version_available(self):
        with mock.patch("urllib.request.urlopen", return_value=self._mock_release("v2.5.0")):
            info = net_query.check_for_update("2.2.4")
        assert info == {"version": "2.5.0",
                        "url": "https://github.com/hd152/originstack/releases/tag/v2.5.0"}

    def test_up_to_date_returns_none(self):
        with mock.patch("urllib.request.urlopen", return_value=self._mock_release("v2.2.4")):
            assert net_query.check_for_update("2.2.4") is None

    def test_current_already_newer_than_latest_returns_none(self):
        """A dev build ahead of the last tagged release should never see itself as outdated."""
        with mock.patch("urllib.request.urlopen", return_value=self._mock_release("v2.2.0")):
            assert net_query.check_for_update("2.2.4") is None

    def test_v_prefix_is_optional_on_either_side(self):
        with mock.patch("urllib.request.urlopen", return_value=self._mock_release("2.5.0")):
            assert net_query.check_for_update("v2.2.4")["version"] == "2.5.0"

    def test_unparseable_current_version_skips_the_request_entirely(self):
        with mock.patch("urllib.request.urlopen") as m:
            assert net_query.check_for_update("dev") is None
        m.assert_not_called()

    def test_malformed_tag_returns_none(self):
        with mock.patch("urllib.request.urlopen", return_value=self._mock_release("nightly-build")):
            assert net_query.check_for_update("2.2.4") is None

    def test_network_error_returns_none(self):
        with mock.patch("urllib.request.urlopen", side_effect=OSError("no network")):
            assert net_query.check_for_update("2.2.4") is None

    def test_bad_json_returns_none(self):
        with mock.patch("urllib.request.urlopen", return_value=_fake_response(b"not json")):
            assert net_query.check_for_update("2.2.4") is None

    def test_respects_offline_mode(self):
        net_query.set_offline(True)
        with mock.patch("urllib.request.urlopen") as m:
            assert net_query.check_for_update("2.2.4") is None
        m.assert_not_called()

    def test_repo_argument_is_used_in_the_request_url(self):
        captured = {}

        def _fake_urlopen(req, timeout=None, context=None):
            captured["url"] = req.full_url
            return self._mock_release("v1.0.0")

        with mock.patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            net_query.check_for_update("0.1.0", repo="someone/fork")
        assert "someone/fork" in captured["url"]


class TestReachability:
    """A server that does not answer the TCP handshake fails fast, once per window."""

    @pytest.fixture
    def check(self, monkeypatch):
        monkeypatch.setattr(net_query, '_HOST_STATE', {})
        monkeypatch.setattr(net_query.urllib.request, 'getproxies', lambda: {})
        return net_query._require_reachable.__wrapped__

    def test_unreachable_host_fails_fast_and_is_remembered(self, check):
        calls = []

        def refuse(addr, timeout=None):
            calls.append((addr, timeout))
            raise OSError('timed out')
        with mock.patch('socket.create_connection', side_effect=refuse):
            for _ in range(3):
                with pytest.raises(net_query.urllib.error.URLError, match='simbad.example'):
                    check('https://simbad.example/sim-tap/sync')
        assert calls == [(('simbad.example', 443), net_query._CONNECT_TIMEOUT)]

    def test_retried_after_the_window(self, check, monkeypatch):
        clock = [1000.0]
        monkeypatch.setattr(net_query.time, 'monotonic', lambda: clock[0])
        with mock.patch('socket.create_connection', side_effect=OSError('down')):
            with pytest.raises(net_query.urllib.error.URLError):
                check('https://a.example/x')
        clock[0] += net_query._UNREACHABLE_TTL + 1
        with mock.patch('socket.create_connection') as ok:
            check('https://a.example/x')
            assert ok.call_count == 1

    def test_a_host_that_answered_is_not_checked_again(self, check):
        with mock.patch('socket.create_connection') as ok:
            check('https://b.example/x')
            check('https://b.example/y')
        assert ok.call_count == 1

    def test_behind_a_proxy_nothing_is_checked(self, check, monkeypatch):
        monkeypatch.setattr(net_query.urllib.request, 'getproxies',
                            lambda: {'https': 'http://proxy:3128'})
        with mock.patch('socket.create_connection', side_effect=OSError('x')) as c:
            check('https://c.example/x')
        assert c.call_count == 0

    def test_offline_wins(self, monkeypatch):
        monkeypatch.setattr(net_query, '_OFFLINE', True)
        with pytest.raises(net_query.OfflineError):
            net_query._require_online('https://d.example/x')

    def test_unreachable_lookup_returns_none_quickly(self, check, monkeypatch):
        monkeypatch.setattr(net_query, '_require_reachable', check)
        import time as _t
        with mock.patch('socket.create_connection', side_effect=OSError('timed out')):
            t0 = _t.monotonic()
            assert net_query.simbad_name_lookup('M42') is None
        assert _t.monotonic() - t0 < 1.0
