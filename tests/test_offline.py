"""--offline: no network request of any kind.

Every HTTP request in the project goes through net_query's three ``_http_*`` helpers, so the
guard sits there and these tests count real ``urlopen`` calls rather than trusting each caller.
"""
import argparse
import urllib.error

import pytest

from src import cli, net_query, target_inference


@pytest.fixture(autouse=True)
def _online_again():
    net_query.set_offline(False)
    yield
    net_query.set_offline(False)


@pytest.fixture
def urlopen_calls(monkeypatch):
    calls = []

    def fake(req, *a, **k):
        calls.append(getattr(req, 'full_url', req))
        raise urllib.error.URLError('no network in tests')
    monkeypatch.setattr(net_query.urllib.request, 'urlopen', fake)
    return calls


def test_offline_blocks_all_three_http_helpers_before_any_socket(urlopen_calls):
    net_query.set_offline(True)
    with pytest.raises(net_query.OfflineError, match='simbad.cds.unistra.fr'):
        net_query._http_get('https://simbad.cds.unistra.fr/x')
    with pytest.raises(net_query.OfflineError):
        net_query._http_post_form('https://gea.esac.esa.int/tap', {'a': '1'})
    with pytest.raises(net_query.OfflineError):
        net_query._http_post_multipart('https://nova.astrometry.net/api/upload', {}, 'file', 'f.fits', b'x')
    assert urlopen_calls == []


def test_online_still_reaches_urlopen(urlopen_calls):
    with pytest.raises(urllib.error.URLError):
        net_query._http_get('https://simbad.cds.unistra.fr/x')
    assert len(urlopen_calls) == 1


def test_every_service_function_is_covered_by_the_guard(urlopen_calls):
    net_query.set_offline(True)
    for call in (lambda: net_query.simbad_name_lookup('M 51'),
                 lambda: net_query.simbad_cone_search(10.0, 20.0),
                 lambda: net_query.gaia_cone_search(10.0, 20.0, 0.5),
                 lambda: net_query.vizier_cone_search(10.0, 20.0, 0.5),
                 lambda: net_query.horizons_ephemeris('C/2025 R2', '2025-10-16T00:00:00')):
        try:
            call()
        except Exception:                        # fail-soft callers may swallow the error
            pass
    assert urlopen_calls == []


def test_unfamiliar_target_name_reaches_simbad_only_when_online(urlopen_calls):
    name = 'Zzyzx Unknownium Nebula 99'
    target_inference.infer_target_from_metadata('nodir', [], use_simbad=True, session_name=name)
    assert urlopen_calls, 'sanity: with the network on, an unknown name is looked up'
    urlopen_calls.clear()
    net_query.set_offline(True)
    target_inference.infer_target_from_metadata('nodir', [], use_simbad=True, session_name=name)
    assert urlopen_calls == []


def test_flag_parses_and_defaults_off():
    import sys
    old = sys.argv
    try:
        sys.argv = ['x', '-d', 'a']
        assert cli.parse_args().offline is False
        sys.argv = ['x', '-d', 'a', '--offline']
        assert cli.parse_args().offline is True
    finally:
        sys.argv = old


def test_policy_turns_online_features_off_and_says_so(capsys):
    args = argparse.Namespace(offline=True, plate_solve=True, annotate=True, photometry=False,
                              photometry_timeseries=False, color_calibrate=True)
    assert cli.apply_network_policy(args, announce=True) is True
    assert net_query.is_offline()
    assert not (args.plate_solve or args.annotate or args.color_calibrate)
    out = capsys.readouterr().out
    assert 'Offline mode' in out and '--plate-solve' in out and '--annotate' in out


def test_policy_is_reset_for_the_next_run_in_the_same_process():
    """The desktop app runs many jobs in one process; --offline must not stick."""
    cli.apply_network_policy(argparse.Namespace(offline=True))
    assert net_query.is_offline()
    cli.apply_network_policy(argparse.Namespace(offline=False))
    assert not net_query.is_offline()
    cli.apply_network_policy(argparse.Namespace())              # flag absent entirely
    assert not net_query.is_offline()
