"""Target inference: catalogue gaps and the "galaxy" keyword fallback.

A galaxy the table doesn't know used to come back type='unknown', so --auto
never skipped sky_residual and the passes fit the galaxy away as background
(real Fireworks Galaxy session).
"""
import pytest

from src.target_inference import infer_target_from_metadata

_NO_DIR = '/nonexistent/qqqqqqqq'


def _infer(name):
    return infer_target_from_metadata(_NO_DIR, [], use_simbad=False, session_name=name)


def test_fireworks_galaxy_is_in_catalogue():
    name, kind, conf, _ = _infer('Fireworks Galaxy')
    assert (name, kind) == ('Fireworks Galaxy', 'galaxy')
    assert conf == 1.0


@pytest.mark.parametrize('name', ['Zork Galaxy', 'the zork galaxies', 'ZORK GALAXY'])
def test_unknown_galaxy_name_falls_back_to_galaxy(name):
    assert _infer(name)[1] == 'galaxy'


@pytest.mark.parametrize('name', ['Zork Cloud', 'Zork Nebula', 'Galaxyless'])
def test_other_unknown_names_stay_unknown(name):
    assert _infer(name)[1] == 'unknown'
