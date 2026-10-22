"""--auto picks the learned denoiser (with chroma NR first) in place of the wavelet denoiser
for non-galaxy targets on Origin data; galaxies keep the learned deconvolver."""
import pytest

from src import auto_settings as a
from src import nn_denoise as nd
from src.cli import parse_args

ORIGIN = {'model': 'Origin178', 'unit': '12345'}
NEBULA = {'emission_nebula': 0.9, 'star_field': 0.1}
GALAXY = {'galaxy': 0.8, 'star_field': 0.2}


def _args(*flags, camera=ORIGIN):
    args = parse_args(['-d', 'x', '-o', 'y.fits', *flags])
    args.denoise_curvelet = True            # what the anchors set before these rules run
    args._camera = camera
    return args


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(nd, 'model_available', lambda path=None: True)


def _both_rules(args, weights):
    return a._apply_nn_deconvolution(args, weights) + a._apply_nn_denoiser(args, weights)


def test_origin_nebula_gets_the_learned_denoiser_with_chroma_nr(model):
    args = _args()
    changes = _both_rules(args, NEBULA)
    assert args.denoise_nn is True and args.denoise_curvelet is False
    assert args.nn_chroma_nr is True
    assert args.deconvolve_nn is False
    assert any(c.startswith('denoise_nn') for c in changes)


def test_galaxy_keeps_the_learned_deconvolver(model):
    args = _args()
    _both_rules(args, GALAXY)
    assert args.deconvolve_nn is True and args.denoise_nn is False


def test_not_where_no_wavelet_denoiser_would_run(model):
    args = _args()
    args.denoise_curvelet = False
    assert a._apply_nn_denoiser(args, NEBULA) == []
    assert args.denoise_nn is False


@pytest.mark.parametrize('camera', [None, {'model': 'Origin678C', 'unit': 'ab12'},
                                    {'model': 'ASI294MC', 'unit': None}])
def test_not_for_other_cameras(model, camera):
    args = _args(camera=camera)
    assert a._apply_nn_denoiser(args, NEBULA) == []
    assert args.denoise_nn is False and args.denoise_curvelet is True


@pytest.mark.parametrize('flags', [['--denoiser', 'wavelet'], ['--denoiser', 'acdnr'],
                                   ['--deconvolve', 'rl'], ['--deconvolve', 'off']])
def test_an_explicit_choice_wins(model, flags):
    args = _args(*flags)
    args.denoise_curvelet = args.denoiser in ('wavelet', 'curvelet') or 'deconvolve' in flags[0]
    before = (args.denoise_nn, args.denoise_curvelet, args.nn_chroma_nr)
    assert a._apply_nn_denoiser(args, NEBULA) == []
    assert (args.denoise_nn, args.denoise_curvelet, args.nn_chroma_nr) == before


def test_not_without_the_model(monkeypatch):
    monkeypatch.setattr(nd, 'model_available', lambda path=None: False)
    args = _args()
    assert a._apply_nn_denoiser(args, NEBULA) == []
    assert args.denoise_nn is False
