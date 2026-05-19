"""Central feature registry for optional dependencies.

Probes all optional dependencies once at import time and exposes boolean
flags (HAS_CV2, HAS_PWT, etc.) so conditional logic throughout the codebase
can reference a single source of truth instead of scattered try/except blocks.
"""
from __future__ import annotations

import logging

_log = logging.getLogger('originstack')

def _probe(name: str, import_func, impact: str):
    """Probe an optional dependency and log if missing."""
    try:
        import_func()
        return True
    except Exception:
        _log.debug("Optional dependency %s not available — %s", name, impact)
        return False


HAS_CV2 = _probe('cv2', lambda: __import__('cv2'),
                  'malvar/vng debayer and bilateral denoise unavailable')
HAS_PWT = _probe('pywt', lambda: __import__('pywt'),
                  'wavelet denoising unavailable')
HAS_PSUTIL = _probe('psutil', lambda: __import__('psutil'),
                     'memory monitoring disabled')
HAS_PHOTUTILS = _probe('photutils',
                        lambda: __import__('photutils.detection', fromlist=['DAOStarFinder']),
                        'star detection uses fallback local-maxima method')
HAS_ASTROPY_STATS = _probe('astropy.stats',
                            lambda: __import__('astropy.stats', fromlist=['sigma_clipped_stats']),
                            'using numpy fallback for clipped statistics')
HAS_SKIMAGE_REGISTRATION = _probe('skimage.registration',
                                   lambda: __import__('skimage.registration', fromlist=['phase_cross_correlation']),
                                   'phase cross-correlation unavailable')
HAS_SKIMAGE_TRANSFORM = _probe('skimage.transform',
                                lambda: (__import__('skimage.transform', fromlist=['EuclideanTransform']),
                                         __import__('skimage.measure', fromlist=['ransac'])),
                                'affine registration unavailable')
HAS_SKIMAGE_EXPOSURE = _probe('skimage.exposure',
                               lambda: __import__('skimage', fromlist=['exposure']),
                               'linear preview stretch unavailable')
HAS_PIL = _probe('Pillow', lambda: __import__('PIL.Image', fromlist=['Image']),
                 'JPEG preview generation disabled')
HAS_CUPY = _probe('cupy', lambda: __import__('cupy'),
                   'GPU acceleration disabled, using CPU')
HAS_ASTROMETRY = _probe('astroquery',
                         lambda: __import__('astroquery.astrometry_net', fromlist=['AstrometryNet']),
                         'plate solving unavailable')
HAS_ANTHROPIC = _probe('anthropic', lambda: __import__('anthropic'),
                        'AI advisor unavailable')
HAS_SCIPY = _probe('scipy', lambda: __import__('scipy.ndimage', fromlist=['ndimage']),
                    'CRITICAL: scipy required for core processing')
HAS_TQDM = _probe('tqdm', lambda: __import__('tqdm', fromlist=['tqdm']),
                   'progress bars disabled')

# --- TOML (Python 3.11+ has tomllib; older versions need tomli) ---
try:
    import tomllib  # noqa: F401
    HAS_TOML = True
except ImportError:
    try:
        import tomli as tomllib  # noqa: F401
        HAS_TOML = True
    except ImportError:
        HAS_TOML = False
        _log.debug("Optional dependency TOML not available — config file support disabled")


def print_feature_status():
    """Print which optional dependencies are available."""
    features = [
        ('OpenCV (cv2)', HAS_CV2, 'malvar/vng debayer, bilateral denoise'),
        ('PyWavelets (pywt)', HAS_PWT, 'wavelet denoising'),
        ('psutil', HAS_PSUTIL, 'memory monitoring'),
        ('photutils', HAS_PHOTUTILS, 'star detection, FWHM measurement'),
        ('scikit-image', HAS_SKIMAGE_REGISTRATION, 'phase correlation, CA correction'),
        ('Pillow (PIL)', HAS_PIL, 'JPEG preview generation'),
        ('CuPy', HAS_CUPY, 'GPU acceleration'),
        ('astroquery', HAS_ASTROMETRY, 'plate solving'),
        ('anthropic', HAS_ANTHROPIC, 'AI advisor & reports'),
        ('TOML', HAS_TOML, 'configuration file support'),
    ]
    for name, available, purpose in features:
        status = 'OK' if available else 'not installed'
        _log.debug("Feature check: %s = %s (%s)", name, status, purpose)
