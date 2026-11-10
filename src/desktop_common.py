"""Desktop-app pieces that do not depend on a GUI toolkit.

Field labels, the "what did you image?" cards, run modes, frame-count text and the
Phase 4 step labels. Shared by the Qt window (``src/qt_app``) and the legacy tkinter one
(``src/desktop_app``), so both show the same wording and translation keys.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from src.i18n import N_, _

#  Shown directly on the main form, in this order -- everything else lives
# behind the "Additional options" toggle. This is the same small set the
# old browser dashboard treated as "quick fields", chosen because they're
# the ones almost every run touches (where the lights are, what to call the
# output, and the handful of settings that most change the result) while
# the other ~110 flags are fine-tuning most runs never need.
_COMMON_DESTS = ['directory', 'output', 'temp_dir', 'auto', 'stack_method',
                 'denoiser', 'deconvolve_mode', 'drizzle_scale', 'trail_reject',
                 'use_gpu', 'parallel']

# Human field labels. Anything not listed falls back to the dest name with
# underscores spaced and the first letter capitalised ("stack_method" ->
# "Stack method"). Only the cases where that reads badly are overridden.
_FIELD_LABELS = {
    'directory': N_('Light frames'),
    'output': N_('Output'),
    'temp_dir': N_('Temp folder'),
    'parallel': N_('Workers'),
    'use_gpu': N_('Use GPU'),
    'trail_reject': N_('Trail rejection'),
    'drizzle_scale': N_('Drizzle scale'),
    'plate_solve': N_('Plate solve'),
    'color_calibrate': N_('Colour calibrate'),
    'remove_stars': N_('Starless sidecar'),
    'deconvolve_mode': N_('Deconvolution'),
    'nn_deconv_blend': N_('Learned sharpening strength'),
    'nn_tilt_equalise': N_('Local optical correction'),
    'session_cfa_eq': N_('Session colour equalisation'),
    'ca_correction': N_('Chromatic aberration fix'),
    'cfa_drizzle': N_('Bayer drizzle'),
    'dark_temp_model': N_('Dark temperature model'),
    'super_res_iters': N_('Super-resolution passes'),
    'bg_method': N_('Background method'),
    'scnr': N_('SCNR (green removal)'),
}


# Groups and fields hidden until "Show expert options" is ticked: fine-tuning
# that --auto manages, diagnostics, and features the project's own testing
# found not to help on typical data. The CLI is unaffected.
_EXPERT_GROUP_PREFIXES = ('Advanced', 'Diagnostics')
_EXPERT_DESTS = {'cfa_drizzle', 'nmf_separate', 'starless_process', 'layered_stretch',
                 'bg_method', 'fix_atmospheric_dispersion', 'distortion_model'}


def _is_expert_group(title: str) -> bool:
    return title.startswith(_EXPERT_GROUP_PREFIXES)


def _is_expert_field(field: Dict[str, Any]) -> bool:
    return field['dest'] in _EXPERT_DESTS or 'experimental' in (field.get('help') or '').lower()


# Required paths: being set is not a change worth marking.
_NO_CHANGE_MARK = {'directory', 'output', 'temp_dir'}


def _field_label(field: Dict[str, Any]) -> str:
    """The English label, which is also its translation key."""
    dest = field['dest']
    return _FIELD_LABELS.get(dest, dest.replace('_', ' ').capitalize())


def _field_label_ui(field: Dict[str, Any]) -> str:
    return _(_field_label(field))

# ── Setup: "what did you image?" cards and the run mode ─────────────

_EXAMPLES_DIR = Path(__file__).resolve().parent / 'data' / 'examples'

# key, title, description, target_type value, comet_mode value, example file,
# example caption. A card is only a view over the target_type/comet_mode
# fields (the same Variables the Additional options widgets use), so a pick
# reaches the run through read_form() like any other change. Descriptions
# say only what auto_settings._TARGET_SETTINGS does for that type.
_TARGET_CARDS = (
    ('auto', N_('Auto-detect'),
     N_('Recognised from the session, header or folder name, then the frames.'),
     '', False, None, None),
    ('galaxy', N_('Galaxy'),
     N_('Keeps the faint halo out of background removal; trims star size.'),
     'galaxy', False, 'galaxy.jpg', N_('Example galaxy: Whirlpool Galaxy (M51), Celestron Origin')),
    ('nebula', N_('Nebula'),
     N_('Diffuse emission: a gentler stretch, stars kept full size.'),
     'emission_nebula', False, 'nebula.jpg', N_('Example nebula: Omega Nebula (M17), Celestron Origin')),
    ('cluster', N_('Star cluster'),
     N_('Dense clusters: a dark sky background, stars kept full size.'),
     'globular_cluster', False, None, None),
    ('starfield', N_('Star field'),
     N_('Rich star fields and Milky Way regions; light local contrast.'),
     'star_field', False, 'starfield.jpg',
     N_('Example star field: Sagittarius Star Cloud (M24), Celestron Origin')),
    ('comet', N_('Comet'),
     N_('Also stacks on the moving nucleus (saved as a second _comet image).'),
     '', True, 'comet.jpg', N_('Example comet: C/2025 A6 (Lemmon), Celestron Origin')),
)
# target_inference's source codes, as the phrase shown after a target name.
_SOURCE_PHRASES = {'session': N_('from the session file'), 'header': N_('from the FITS header'),
                   'folder': N_('from the folder name'), 'filename': N_('from the file name'),
                   'simbad': N_('from SIMBAD')}


def _from_source(src: Optional[str]) -> str:
    return _(_SOURCE_PHRASES[src]) if src in _SOURCE_PHRASES else ''


def _target_label(target_type: str) -> str:
    from src.auto_settings import TARGET_LABELS
    return _(TARGET_LABELS.get(target_type, target_type))


# Inferred target types -> the card that shows them.
_TYPE_TO_CARD = {'galaxy': 'galaxy', 'emission_nebula': 'nebula', 'reflection_nebula': 'nebula',
                 'planetary_nebula': 'nebula', 'globular_cluster': 'cluster',
                 'star_field': 'starfield', 'wide_field': 'starfield'}

# key, title, description, preset value. Only two: the default settings are
# the measured best, and --preset quality forces per-frame L.A.Cosmic (~2.5
# min for <0.3% change) and deconvolution (no help at typical Origin SNR), so
# offering it as "best quality" would sell a slower run, not a better one.
_RUN_MODES = (
    ('full', N_('Full quality'), N_('Recommended. Every step that measured as an improvement.'), ''),
    ('quick', N_('Quick look'), N_('Faster check: no outlier rejection, lighter processing.'), 'quick'),
)
# Fields the cards and run mode set; not counted as "changed" settings.
_PICKER_DESTS = {'target_type', 'comet_mode', 'preset'}


# discover_frames() kinds, singular and plural.
_FRAME_KINDS = {'light': (N_('{n} light'), N_('{n} lights')),
                'dark': (N_('{n} dark'), N_('{n} darks')),
                'flat': (N_('{n} flat'), N_('{n} flats')),
                'bias': (N_('{n} bias'), N_('{n} bias frames')),
                'darkflat': (N_('{n} dark flat'), N_('{n} dark flats'))}


def with_export_checkboxes(form: Dict[str, Any], typed: str, ticked: Dict[str, bool]) -> Dict[str, Any]:
    """``form`` with ``export`` = the formats typed under Additional options plus the ticked
    "Also save" boxes (png, png16), each once, in that order. Unchanged when nothing is
    ticked, so an untouched form still leaves --export to its default."""
    extra = [f for f, on in ticked.items() if on]
    if not extra:
        return form
    fmts = [e.strip() for e in str(form.get('export', typed) or '').split(',') if e.strip()]
    fmts += [f for f in extra if f not in fmts]
    return {**form, 'export': ','.join(fmts)}


def _frame_count(kind: str, n: int) -> str:
    one, many = _FRAME_KINDS.get(kind, ('{n} ' + kind, '{n} ' + kind))
    return _(one if n == 1 else many, n=n)


# Labels of the Phase 4 steps the Post-processing panel can switch (postprocess.PHASE4_TOGGLES),
# marked here so the catalogs pick them up; the panel translates them where it shows them.
_PP_STEP_LABELS = {
    'hot_pixel': N_('Hot pixels'),
    'background': N_('Background extraction'),
    'chroma_nr': N_('Chroma noise reduction'),
    'bilateral': N_('Bilateral denoise'),
    'acdnr': N_('ACDNR denoise'),
    'wavelet': N_('Wavelet denoise'),
    'nn': N_('Neural-network denoise / deconvolution'),
    'aniso': N_('Anisotropic diffusion'),
    'scnr': N_('SCNR'),
    'photo_cal': N_('Gray-locus colour calibration'),
    'deconvolve': N_('Deconvolution'),
    'star_reduce': N_('Star reduction'),
    'local_contrast': N_('Local contrast'),
    'sky_neutralize': N_('Final sky flatten'),
    'edge_bands': N_('Edge bands'),
}
