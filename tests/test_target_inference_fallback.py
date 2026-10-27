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


# Names the Origin app writes into info.json (objectName), from a 77-target archive.
@pytest.mark.parametrize('name, kind', [
    ('Duck Nebula', 'emission_nebula'),        # NGC 2359; SIMBAD answers a sub-mm source
    ('Cone Nebula', 'emission_nebula'),
    ('Pacman Nebula', 'emission_nebula'),
    ('Medusa Nebula', 'planetary_nebula'),
    ('Blue Flash Nebula', 'planetary_nebula'),
    ('Fornax A', 'galaxy'),
    ("Markarian's Chain", 'galaxy'),
    ("Stephan's Quintet", 'galaxy'),
    ("Bode's Nebulae", 'galaxy'),
    ('Intergalactic Wanderer', 'globular_cluster'),
    ('Epsilon Orionis Nebula', 'reflection_nebula'),
    ("Ptolemy's Cluster", 'star_field'),
    ('Messier 101', 'galaxy'),                 # 'Messier N' is a catalogue ID
    ('Messier 95', 'galaxy'),
])
def test_origin_object_names_resolve_offline(name, kind):
    assert _infer(name)[1] == kind


@pytest.mark.parametrize('otype, kind', [
    ('GiP', 'galaxy'), ('G', 'galaxy'), ('rG', 'galaxy'), ('Sy2', 'galaxy'), ('CGG', 'galaxy'),
    ('Bla', 'galaxy'), ('G?', 'galaxy'), ('PN', 'planetary_nebula'), ('GlC', 'globular_cluster'),
    ('HII', 'emission_nebula'), ('SNR', 'emission_nebula'), ('RNe', 'reflection_nebula'),
    ('OpC', 'star_field'), ('smm', 'unknown'),
])
def test_simbad_short_otype_codes(otype, kind):
    from src.target_inference import _simbad_type_to_internal
    assert _simbad_type_to_internal(otype) == kind


@pytest.mark.parametrize("name", [
    "C/2025 R2 (SWAN)", "12P/Pons-Brooks", "3I/ATLAS", "C/2023 A3",
    "C2025_R2_(SWAN)_2025-10-20_20-47-16", "3IATLAS_2025-12-18_23-25-37", "C2024_E1_(Wierzchos)"])
def test_comet_designations_are_recognised(name):
    from src.target_inference import comet_designation
    assert comet_designation(None, name) == name


@pytest.mark.parametrize("name", [
    "M 31", "NGC 7000", "Messier_3_2026-07-22_20-53-46", "Cetus A", "Caldwell 14", "IC 434",
    "Sh2-155", "3C 273", "Crab_Nebula_2026-01-26", "Comet"])
def test_deep_sky_names_are_not_comets(name):
    from src.target_inference import comet_designation
    assert comet_designation(name) is None


def test_auto_turns_comet_mode_on_for_a_comet_unless_told_not_to():
    from src.auto_settings import _apply_comet_mode
    from src.cli import parse_args
    a = parse_args(['-d', '.'])
    a._inferred_comet = 'C/2025 R2 (SWAN)'
    assert _apply_comet_mode(a) and a.comet_mode is True
    b = parse_args(['-d', '.', '--no-comet-mode'])
    b._inferred_comet = 'C/2025 R2 (SWAN)'
    assert _apply_comet_mode(b) == [] and b.comet_mode is False
    c = parse_args(['-d', '.'])
    c._inferred_comet = None
    assert _apply_comet_mode(c) == [] and c.comet_mode is False
