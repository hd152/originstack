"""--auto picks the learned deconvolver (--deconvolve nn): galaxies at their own blend, other targets
where the wavelet denoiser would run. Off Origin it is tentative (guards in Phase 4)."""
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
def test_other_targets_get_it_at_the_other_blend(model, weights):
    args = _args()
    a._apply_nn_deconvolution(args, weights)
    assert args.deconvolve_nn is True and args.denoise_curvelet is False
    assert args.nn_deconv_blend == a.NN_DECONV_OTHER_BLEND


def test_other_targets_only_where_the_wavelet_denoiser_would_run(model):
    args = _args()
    args.denoise_curvelet = False
    assert a._apply_nn_deconvolution(args, {'emission_nebula': 1.0}) == []
    assert args.deconvolve_nn is False


# Cameras the network was not trained on (the Mark II's Origin678C, Seestar, ASI...) get it too,
# tentatively: the stack guards in Phase 4 (auto_settings.nn_stack_guard) can still take it back
@pytest.mark.parametrize('camera', [None, {'model': 'IMX571', 'unit': None},
                                    {'model': 'Origin678C', 'unit': 'ab12', 'profiled': False},
                                    {'model': 'ASI294MC', 'unit': None}])
def test_other_cameras_get_it_with_a_fallback(model, camera):
    args = _args(camera=camera)
    before = (args.deconvolve_mode, args.deconvolve_nn, args.nn_deconv_blend, args.denoise_curvelet)
    a._apply_nn_deconvolution(args, GALAXY, n_frames=50)
    assert args.deconvolve_nn is True and args.nn_deconv_blend == a.NN_DECONV_GALAXY_BLEND
    fb = args._auto_nn_fallback                      # what Phase 4 restores if a guard trips
    assert (fb['deconvolve_mode'], fb['deconvolve_nn'], fb['nn_deconv_blend'], fb['denoise_curvelet']) == before


def test_origin178_keeps_no_fallback(model):
    args = _args()
    a._apply_nn_deconvolution(args, GALAXY, n_frames=2)      # even a short session: unchanged rule
    assert args.deconvolve_nn is True and not hasattr(args, '_auto_nn_fallback')


def test_off_origin_needs_a_stack_of_frames(model):
    args = _args(camera={'model': 'ASI294MC', 'unit': None})
    assert a._apply_nn_deconvolution(args, GALAXY, n_frames=a.NN_OFFORIGIN_MIN_FRAMES - 1) == []
    assert args.deconvolve_nn is False and not hasattr(args, '_auto_nn_fallback')
    a._apply_nn_deconvolution(args, GALAXY, n_frames=a.NN_OFFORIGIN_MIN_FRAMES)
    assert args.deconvolve_nn is True


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
    monkeypatch.setattr(a, '_apply_nn_deconvolution', lambda args, weights, n_frames=None: seen.append(weights) or [])
    monkeypatch.setattr(a, '_aggregate', lambda final: {})
    monkeypatch.setattr(a, '_compute_signals', lambda agg: {})
    monkeypatch.setattr(a, '_classify', lambda sig: 'galaxy')
    monkeypatch.setattr(a, '_blend_weights', lambda sig, **k: dict(GALAXY))
    monkeypatch.setattr(a, '_apply_dynamic_settings', lambda sig, weights, args: [])
    monkeypatch.setattr(a, '_apply_quality_settings', lambda sig, args, **k: [])
    monkeypatch.setattr(a, '_apply_free_diagnostics', lambda args: [])
    a.apply_auto_settings([object()], _args())
    assert seen == [GALAXY]


def test_galaxies_get_the_galaxy_blend(model):
    args = _args()
    assert args.nn_deconv_blend is None                           # automatic
    a._apply_nn_deconvolution(args, GALAXY)
    assert args.nn_deconv_blend == a.NN_DECONV_GALAXY_BLEND


def test_a_configured_blend_wins(model):
    args = _args()
    args.nn_deconv_blend = 0.5                  # e.g. from --config
    a._apply_nn_deconvolution(args, GALAXY)
    assert args.nn_deconv_blend == 0.5


def test_nn_deconv_blend_flag_default_is_automatic_and_validated(model):
    assert _args().nn_deconv_blend is None
    assert _args('--nn-deconv-blend', '0.6').nn_deconv_blend == 0.6
    for bad in ('1.5', '-0.1', 'x'):
        with pytest.raises(SystemExit):
            _args('--nn-deconv-blend', bad)


def test_an_explicit_blend_flag_beats_auto_even_at_the_old_default(model):
    for value in (0.75, 0.5):
        args = _args('--nn-deconv-blend', str(value))
        a._apply_nn_deconvolution(args, GALAXY)
        assert args.nn_deconv_blend == value


def test_blend_flag_in_form_schema_and_saved_config_roundtrip(model, tmp_path):
    from src.cli import save_effective_config
    from src.desktop_control import get_form_schema
    fields = [f for g in get_form_schema().values() for f in g]
    assert any(f['dest'] == 'nn_deconv_blend' and f['default'] is None for f in fields)
    args = _args('--nn-deconv-blend', '0.4')
    out = str(tmp_path / 's.fits')
    save_effective_config(args, out)
    assert 'nn_deconv_blend = 0.4' in (tmp_path / 's_config.toml').read_text()
    save_effective_config(_args(), out)
    assert 'nn_deconv_blend' not in (tmp_path / 's_config.toml').read_text()
