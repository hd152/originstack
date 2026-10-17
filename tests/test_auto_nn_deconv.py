"""--auto picks the learned deconvolver (--deconvolve nn) for galaxies on Origin data only."""
import pytest

from src import auto_settings as a
from src import nn_denoise as nd
from src.cli import parse_args

ORIGIN = {'model': 'Origin178', 'unit': '12345'}
GALAXY = {'galaxy': 0.8, 'star_field': 0.2}


def _args(*flags, camera=ORIGIN):
    args = parse_args(['-d', 'x', '-o', 'y.fits', *flags])
    args.denoise_curvelet = True            # what the galaxy anchor sets before this rule runs
    args._camera = camera
    return args


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(nd, 'model_available', lambda path=None: True)


def test_origin_galaxy_gets_the_learned_deconvolver(model):
    args = _args()
    changes = a._apply_nn_deconvolution(args, GALAXY)
    assert args.deconvolve_nn is True and args.deconvolve_mode == 'nn'
    assert args.deconvolve is False
    assert not (args.denoise_curvelet or args.denoise_acdnr or args.denoise_bilateral
                or args.denoise_aniso or args.denoise_nn)
    assert any(c.startswith('deconvolve_nn') for c in changes)


@pytest.mark.parametrize('weights', [{'galaxy': 0.45, 'emission_nebula': 0.55},
                                     {'emission_nebula': 1.0}, {}])
def test_not_for_other_targets(model, weights):
    args = _args()
    assert a._apply_nn_deconvolution(args, weights) == []
    assert args.deconvolve_nn is False and args.denoise_curvelet is True


# the Mark II's Origin678C and the other cameras the Origin platform drives were not in
# the training data: 'starts with Origin' is not enough
@pytest.mark.parametrize('camera', [None, {'model': 'IMX571', 'unit': None},
                                    {'model': 'Origin678C', 'unit': 'ab12', 'profiled': False},
                                    {'model': 'ASI294MC', 'unit': None}])
def test_not_for_other_cameras(model, camera):
    args = _args(camera=camera)
    assert a._apply_nn_deconvolution(args, GALAXY) == []
    assert args.deconvolve_nn is False


@pytest.mark.parametrize('flags', [['--denoiser', 'wavelet'], ['--deconvolve', 'rl'],
                                   ['--deconvolve', 'off'], ['--denoiser', 'nn']])
def test_an_explicit_choice_wins(model, flags):
    args = _args(*flags)
    before = (args.deconvolve_mode, args.deconvolve_nn, args.denoise_nn)
    assert a._apply_nn_deconvolution(args, GALAXY) == []
    assert (args.deconvolve_mode, args.deconvolve_nn, args.denoise_nn) == before


def test_not_without_the_model(monkeypatch):
    monkeypatch.setattr(nd, 'model_available', lambda path=None: False)
    args = _args()
    assert a._apply_nn_deconvolution(args, GALAXY) == []
    assert args.deconvolve_nn is False


def test_runs_as_part_of_apply_auto_settings(model, monkeypatch):
    seen = []
    monkeypatch.setattr(a, '_apply_nn_deconvolution', lambda args, weights: seen.append(weights) or [])
    monkeypatch.setattr(a, '_aggregate', lambda final: {})
    monkeypatch.setattr(a, '_compute_signals', lambda agg: {})
    monkeypatch.setattr(a, '_classify', lambda sig: 'galaxy')
    monkeypatch.setattr(a, '_blend_weights', lambda sig, **k: dict(GALAXY))
    monkeypatch.setattr(a, '_apply_dynamic_settings', lambda sig, weights, args: [])
    monkeypatch.setattr(a, '_apply_quality_settings', lambda sig, args, **k: [])
    monkeypatch.setattr(a, '_apply_free_diagnostics', lambda args: [])
    a.apply_auto_settings([object()], _args())
    assert seen == [GALAXY]
