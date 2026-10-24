"""Galaxy previews put the black point at the sky median, not above it: at 1 sigma above (the
old preset) 27-57% of the faint galaxy pixels rendered pure black once the learned
deconvolver had removed most of the sky noise."""
from src.auto_settings import _TARGET_SETTINGS


def test_galaxy_preset_black_point_is_the_sky_median():
    assert dict(_TARGET_SETTINGS['galaxy'])['preview_black_sigma'] == 0.0
