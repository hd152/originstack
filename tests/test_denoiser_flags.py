"""--denoiser wavelet/curvelet are one denoiser; protection is --wavelet-protect."""
import pytest

from src.cli import parse_args


def _parse(*extra):
    return parse_args(['-d', 'x', '-o', 'y.fits', *extra])


def test_default_protection_is_unchanged():
    a = _parse()
    assert a.denoiser == 'auto' and a.directional_protect_strength == 0.6


@pytest.mark.parametrize('name', ['wavelet', 'curvelet'])
def test_wavelet_and_its_curvelet_alias_are_identical(name):
    a = _parse('--denoiser', name)
    assert a.denoiser == 'wavelet'                 # alias normalised
    assert a.denoise_curvelet is True
    assert a.directional_protect_strength == 0.6   # protection NOT forced to 0 any more
    assert 'denoise_curvelet' in a._explicit_cli_dests


def test_protection_is_its_own_option_and_counts_as_explicit():
    a = _parse('--denoiser', 'wavelet', '--wavelet-protect', '0')
    assert a.directional_protect_strength == 0.0
    assert 'directional_protect_strength' in a._explicit_cli_dests
    assert 'directional_protect_strength' not in _parse('--denoiser', 'wavelet')._explicit_cli_dests


@pytest.mark.parametrize('bad', ['1.5', '-0.1', 'x'])
def test_protection_must_be_a_number_in_0_1(bad, capsys):
    with pytest.raises(SystemExit):
        _parse('--wavelet-protect', bad)
    capsys.readouterr()


def test_other_denoisers_turn_the_wavelet_step_off():
    a = _parse('--denoiser', 'acdnr')
    assert a.denoise_curvelet is False and a.denoise_acdnr is True
