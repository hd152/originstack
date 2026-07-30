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
