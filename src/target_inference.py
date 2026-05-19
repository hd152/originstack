"""Infer the astrophotography target from FITS headers, folder names, and file names.

Priority order (highest to lowest confidence):
  1. FITS OBJECT keyword  — written by capture software; most reliable.
  2. Folder name          — strip trailing date/time, replace underscores → spaces.
  3. File names           — catalogue-ID regex only; common abbreviations are ignored.
  4. Simbad query         — optional network fallback via astroquery.

Public API
----------
infer_target_from_metadata(directory, frames, use_simbad=True)
    -> (target_name, object_type, confidence, source)
    target_name : str | None  — canonical or common name
    object_type : str | None  — one of the auto_settings type keys
    confidence  : float       — 0.0 – 1.0
    source      : str | None  — 'header' | 'folder' | 'filename' | 'simbad'
"""
from __future__ import annotations

import logging
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static catalogue
# ---------------------------------------------------------------------------
# Format: normalised_id → (common_name, object_type)
# object_type keys must match TARGET_LABELS in auto_settings.py.

_G  = 'galaxy'
_GL = 'globular_cluster'
_PN = 'planetary_nebula'
_EN = 'emission_nebula'
_RN = 'reflection_nebula'
_SF = 'star_field'
_WF = 'wide_field'

_CATALOGUE: Dict[str, Tuple[str, str]] = {
    # ----- Messier objects -----
    'M1':   ('Crab Nebula',                  _EN),
    'M2':   ('Messier 2',                    _GL),
    'M3':   ('Messier 3',                    _GL),
    'M4':   ('Messier 4',                    _GL),
    'M5':   ('Messier 5',                    _GL),
    'M6':   ('Butterfly Cluster',            _SF),
    'M7':   ('Ptolemy Cluster',              _SF),
    'M8':   ('Lagoon Nebula',                _EN),
    'M9':   ('Messier 9',                    _GL),
    'M10':  ('Messier 10',                   _GL),
    'M11':  ('Wild Duck Cluster',            _SF),
    'M12':  ('Messier 12',                   _GL),
    'M13':  ('Great Hercules Cluster',       _GL),
    'M14':  ('Messier 14',                   _GL),
    'M15':  ('Messier 15',                   _GL),
    'M16':  ('Eagle Nebula',                 _EN),
    'M17':  ('Omega Nebula',                 _EN),
    'M18':  ('Messier 18',                   _SF),
    'M19':  ('Messier 19',                   _GL),
    'M20':  ('Trifid Nebula',                _EN),
    'M21':  ('Messier 21',                   _SF),
    'M22':  ('Great Sagittarius Cluster',    _GL),
    'M23':  ('Messier 23',                   _SF),
    'M24':  ('Sagittarius Star Cloud',       _WF),
    'M25':  ('Messier 25',                   _SF),
    'M26':  ('Messier 26',                   _SF),
    'M27':  ('Dumbbell Nebula',              _PN),
    'M28':  ('Messier 28',                   _GL),
    'M29':  ('Messier 29',                   _SF),
    'M30':  ('Messier 30',                   _GL),
    'M31':  ('Andromeda Galaxy',             _G),
    'M32':  ('Messier 32',                   _G),
    'M33':  ('Triangulum Galaxy',            _G),
    'M34':  ('Messier 34',                   _SF),
    'M35':  ('Messier 35',                   _SF),
    'M36':  ('Messier 36',                   _SF),
    'M37':  ('Messier 37',                   _SF),
    'M38':  ('Messier 38',                   _SF),
    'M39':  ('Messier 39',                   _SF),
    'M40':  ('Winnecke 4',                   _SF),
    'M41':  ('Messier 41',                   _SF),
    'M42':  ('Orion Nebula',                 _EN),
    'M43':  ("De Mairan's Nebula",           _EN),
    'M44':  ('Beehive Cluster',              _SF),
    'M45':  ('Pleiades',                     _RN),
    'M46':  ('Messier 46',                   _SF),
    'M47':  ('Messier 47',                   _SF),
    'M48':  ('Messier 48',                   _SF),
    'M49':  ('Messier 49',                   _G),
    'M50':  ('Messier 50',                   _SF),
    'M51':  ('Whirlpool Galaxy',             _G),
    'M52':  ('Messier 52',                   _SF),
    'M53':  ('Messier 53',                   _GL),
    'M54':  ('Messier 54',                   _GL),
    'M55':  ('Messier 55',                   _GL),
    'M56':  ('Messier 56',                   _GL),
    'M57':  ('Ring Nebula',                  _PN),
    'M58':  ('Messier 58',                   _G),
    'M59':  ('Messier 59',                   _G),
    'M60':  ('Messier 60',                   _G),
    'M61':  ('Messier 61',                   _G),
    'M62':  ('Messier 62',                   _GL),
    'M63':  ('Sunflower Galaxy',             _G),
    'M64':  ('Black Eye Galaxy',             _G),
    'M65':  ('Messier 65',                   _G),
    'M66':  ('Messier 66',                   _G),
    'M67':  ('Messier 67',                   _SF),
    'M68':  ('Messier 68',                   _GL),
    'M69':  ('Messier 69',                   _GL),
    'M70':  ('Messier 70',                   _GL),
    'M71':  ('Messier 71',                   _GL),
    'M72':  ('Messier 72',                   _GL),
    'M73':  ('Messier 73',                   _SF),
    'M74':  ('Phantom Galaxy',               _G),
    'M75':  ('Messier 75',                   _GL),
    'M76':  ('Little Dumbbell Nebula',       _PN),
    'M77':  ('Cetus A',                      _G),
    'M78':  ('Messier 78',                   _RN),
    'M79':  ('Messier 79',                   _GL),
    'M80':  ('Messier 80',                   _GL),
    'M81':  ("Bode's Galaxy",                _G),
    'M82':  ('Cigar Galaxy',                 _G),
    'M83':  ('Southern Pinwheel Galaxy',     _G),
    'M84':  ('Messier 84',                   _G),
    'M85':  ('Messier 85',                   _G),
    'M86':  ('Messier 86',                   _G),
    'M87':  ('Virgo A',                      _G),
    'M88':  ('Messier 88',                   _G),
    'M89':  ('Messier 89',                   _G),
    'M90':  ('Messier 90',                   _G),
    'M91':  ('Messier 91',                   _G),
    'M92':  ('Messier 92',                   _GL),
    'M93':  ('Messier 93',                   _SF),
    'M94':  ("Croc's Eye Galaxy",            _G),
    'M95':  ('Messier 95',                   _G),
    'M96':  ('Messier 96',                   _G),
    'M97':  ('Owl Nebula',                   _PN),
    'M98':  ('Messier 98',                   _G),
    'M99':  ('Coma Pinwheel Galaxy',         _G),
    'M100': ('Mirror Galaxy',                _G),
    'M101': ('Pinwheel Galaxy',              _G),
    'M102': ('Spindle Galaxy',               _G),
    'M103': ('Messier 103',                  _SF),
    'M104': ('Sombrero Galaxy',              _G),
    'M105': ('Messier 105',                  _G),
    'M106': ('Messier 106',                  _G),
    'M107': ('Messier 107',                  _GL),
    'M108': ('Surfboard Galaxy',             _G),
    'M109': ('Messier 109',                  _G),
    'M110': ('Messier 110',                  _G),

    # ----- NGC objects -----
    'NGC104':  ('47 Tucanae',                _GL),
    'NGC224':  ('Andromeda Galaxy',          _G),
    'NGC253':  ('Sculptor Galaxy',           _G),
    'NGC292':  ('Small Magellanic Cloud',    _WF),
    'NGC869':  ('h Persei',                  _SF),
    'NGC884':  ('Chi Persei',                _SF),
    'NGC891':  ('NGC 891',                   _G),
    'NGC1499': ('California Nebula',         _EN),
    'NGC1502': ('NGC 1502',                  _SF),
    'NGC1579': ('NGC 1579',                  _RN),
    'NGC1976': ('Orion Nebula',              _EN),
    'NGC2024': ('Flame Nebula',              _EN),
    'NGC2070': ('Tarantula Nebula',          _EN),
    'NGC2237': ('Rosette Nebula',            _EN),
    'NGC2238': ('Rosette Nebula',            _EN),
    'NGC2239': ('Rosette Nebula',            _EN),
    'NGC2244': ('Rosette Cluster',           _SF),
    'NGC2246': ('Rosette Nebula',            _EN),
    'NGC2392': ('Eskimo Nebula',             _PN),
    'NGC2403': ('NGC 2403',                  _G),
    'NGC2683': ('NGC 2683',                  _G),
    'NGC2903': ('NGC 2903',                  _G),
    'NGC3031': ("Bode's Galaxy",             _G),
    'NGC3034': ('Cigar Galaxy',              _G),
    'NGC3372': ('Carina Nebula',             _EN),
    'NGC3628': ('Hamburger Galaxy',          _G),
    'NGC3718': ('NGC 3718',                  _G),
    'NGC4038': ('Antennae Galaxies',         _G),
    'NGC4039': ('Antennae Galaxies',         _G),
    'NGC4258': ('Messier 106',               _G),
    'NGC4449': ('NGC 4449',                  _G),
    'NGC4565': ('Needle Galaxy',             _G),
    'NGC4631': ('Whale Galaxy',              _G),
    'NGC4889': ('NGC 4889',                  _G),
    'NGC5128': ('Centaurus A',               _G),
    'NGC5139': ('Omega Centauri',            _GL),
    'NGC5194': ('Whirlpool Galaxy',          _G),
    'NGC5236': ('Southern Pinwheel Galaxy',  _G),
    'NGC5907': ('Splinter Galaxy',           _G),
    'NGC6188': ('NGC 6188',                  _EN),
    'NGC6231': ('NGC 6231',                  _SF),
    'NGC6334': ("Cat's Paw Nebula",          _EN),
    'NGC6357': ('War and Peace Nebula',      _EN),
    'NGC6543': ("Cat's Eye Nebula",          _PN),
    'NGC6618': ('Omega Nebula',              _EN),
    'NGC6720': ('Ring Nebula',               _PN),
    'NGC6888': ('Crescent Nebula',           _EN),
    'NGC6960': ('Western Veil Nebula',       _EN),
    'NGC6992': ('Eastern Veil Nebula',       _EN),
    'NGC7000': ('North America Nebula',      _EN),
    'NGC7009': ('Saturn Nebula',             _PN),
    'NGC7023': ('Iris Nebula',               _RN),
    'NGC7027': ('NGC 7027',                  _PN),
    'NGC7089': ('Messier 2',                 _GL),
    'NGC7293': ('Helix Nebula',              _PN),
    'NGC7331': ('NGC 7331',                  _G),
    'NGC7380': ('Wizard Nebula',             _EN),
    'NGC7662': ('Blue Snowball',             _PN),

    # ----- IC objects -----
    'IC342':  ('IC 342',                     _G),
    'IC405':  ('Flaming Star Nebula',        _RN),
    'IC410':  ('Tadpoles Nebula',            _EN),
    'IC434':  ('Horsehead Nebula',           _EN),
    'IC443':  ('Jellyfish Nebula',           _EN),
    'IC1295': ('IC 1295',                    _PN),
    'IC1396': ("Elephant's Trunk Nebula",    _EN),
    'IC1805': ('Heart Nebula',               _EN),
    'IC1848': ('Soul Nebula',                _EN),
    'IC2118': ('Witch Head Nebula',          _RN),
    'IC2177': ('Seagull Nebula',             _EN),
    'IC4406': ('Retina Nebula',              _PN),
    'IC4628': ('Prawn Nebula',               _EN),
    'IC5067': ('Pelican Nebula',             _EN),
    'IC5070': ('Pelican Nebula',             _EN),
    'IC5146': ('Cocoon Nebula',              _EN),

    # ----- Sharpless -----
    'SH2-101': ('Tulip Nebula',              _EN),
    'SH2-106': ('SH2-106',                   _EN),
    'SH2-132': ('Lion Nebula',               _EN),
    'SH2-140': ('SH2-140',                   _EN),
    'SH2-155': ('Cave Nebula',               _EN),
    'SH2-188': ('SH2-188',                   _EN),
    'SH2-240': ('Spaghetti Nebula',          _EN),
    'SH2-252': ('Monkey Head Nebula',        _EN),
    'SH2-261': ("Lower's Nebula",            _EN),
    'SH2-264': ('Lambda Orionis Ring',       _EN),
    'SH2-279': ('Running Man Nebula',        _RN),
    'SH2-308': ('Dolphin Nebula',            _EN),
}

# ---------------------------------------------------------------------------
# Common-name lookup
# ---------------------------------------------------------------------------
# Keys are lowercase; values are (catalogue_id, object_type).
# Catalogue IDs here must be keys in _CATALOGUE (for name retrieval) or
# can be None when only the type matters.

_COMMON_NAMES: Dict[str, Tuple[Optional[str], str]] = {
    # Galaxies
    'andromeda galaxy':        ('M31',     _G),
    'andromeda':               ('M31',     _G),
    'triangulum galaxy':       ('M33',     _G),
    'pinwheel galaxy':         ('M101',    _G),
    'whirlpool galaxy':        ('M51',     _G),
    'whirlpool':               ('M51',     _G),
    'sunflower galaxy':        ('M63',     _G),
    'black eye galaxy':        ('M64',     _G),
    "bode's galaxy":           ('M81',     _G),
    'bodes galaxy':            ('M81',     _G),
    'cigar galaxy':            ('M82',     _G),
    'southern pinwheel galaxy':('M83',     _G),
    'southern pinwheel':       ('M83',     _G),
    'phantom galaxy':          ('M74',     _G),
    'sombrero galaxy':         ('M104',    _G),
    'sombrero':                ('M104',    _G),
    'coma pinwheel galaxy':    ('M99',     _G),
    'mirror galaxy':           ('M100',    _G),
    'surfboard galaxy':        ('M108',    _G),
    'spindle galaxy':          ('M102',    _G),
    'sculptor galaxy':         ('NGC253',  _G),
    'needle galaxy':           ('NGC4565', _G),
    'whale galaxy':            ('NGC4631', _G),
    'hamburger galaxy':        ('NGC3628', _G),
    'centaurus a':             ('NGC5128', _G),
    'antennae galaxies':       ('NGC4038', _G),
    'antennae':                ('NGC4038', _G),
    'leo triplet':             ('M65',     _G),
    'virgo a':                 ('M87',     _G),
    'splinter galaxy':         ('NGC5907', _G),
    "croc's eye galaxy":       ('M94',     _G),

    # Globular clusters
    'great hercules cluster':  ('M13',    _GL),
    'hercules cluster':        ('M13',    _GL),
    'great sagittarius cluster':('M22',   _GL),
    'omega centauri':          ('NGC5139',_GL),
    '47 tucanae':              ('NGC104', _GL),
    '47 tuc':                  ('NGC104', _GL),
    'wild duck cluster':       ('M11',    _GL),

    # Planetary nebulae
    'ring nebula':             ('M57',    _PN),
    'dumbbell nebula':         ('M27',    _PN),
    'little dumbbell nebula':  ('M76',    _PN),
    'owl nebula':              ('M97',    _PN),
    'helix nebula':            ('NGC7293',_PN),
    'eye of god':              ('NGC7293',_PN),
    "cat's eye nebula":        ('NGC6543',_PN),
    'eskimo nebula':           ('NGC2392',_PN),
    'clownface nebula':        ('NGC2392',_PN),
    'saturn nebula':           ('NGC7009',_PN),
    'blue snowball':           ('NGC7662',_PN),
    'retina nebula':           ('IC4406', _PN),

    # Emission nebulae
    'crab nebula':             ('M1',     _EN),
    'lagoon nebula':           ('M8',     _EN),
    'trifid nebula':           ('M20',    _EN),
    'eagle nebula':            ('M16',    _EN),
    'pillars of creation':     ('M16',    _EN),
    'omega nebula':            ('M17',    _EN),
    'swan nebula':             ('M17',    _EN),
    'orion nebula':            ('M42',    _EN),
    'great nebula in orion':   ('M42',    _EN),
    'horsehead nebula':        ('IC434',  _EN),
    'horsehead':               ('IC434',  _EN),
    'flame nebula':            ('NGC2024',_EN),
    'rosette nebula':          ('NGC2237',_EN),
    'north america nebula':    ('NGC7000',_EN),
    'north american nebula':   ('NGC7000',_EN),
    'veil nebula':             ('NGC6992',_EN),
    'eastern veil nebula':     ('NGC6992',_EN),
    'eastern veil':            ('NGC6992',_EN),
    'western veil nebula':     ('NGC6960',_EN),
    'western veil':            ('NGC6960',_EN),
    "witch's broom":           ('NGC6960',_EN),
    'california nebula':       ('NGC1499',_EN),
    'heart nebula':            ('IC1805', _EN),
    'soul nebula':             ('IC1848', _EN),
    'heart and soul':          ('IC1805', _EN),
    'tadpoles nebula':         ('IC410',  _EN),
    "elephant's trunk nebula": ('IC1396', _EN),
    "elephant's trunk":        ('IC1396', _EN),
    'pelican nebula':          ('IC5070', _EN),
    'cocoon nebula':           ('IC5146', _EN),
    'seagull nebula':          ('IC2177', _EN),
    'jellyfish nebula':        ('IC443',  _EN),
    'tarantula nebula':        ('NGC2070',_EN),
    '30 doradus':              ('NGC2070',_EN),
    'carina nebula':           ('NGC3372',_EN),
    'eta carinae nebula':      ('NGC3372',_EN),
    'cat\'s paw nebula':       ('NGC6334',_EN),
    'war and peace nebula':    ('NGC6357',_EN),
    'crescent nebula':         ('NGC6888',_EN),
    'wizard nebula':           ('NGC7380',_EN),
    'lion nebula':             ('SH2-132',_EN),
    'cave nebula':             ('SH2-155',_EN),
    'spaghetti nebula':        ('SH2-240',_EN),
    'tulip nebula':            ('SH2-101',_EN),
    'monkey head nebula':      ('SH2-252',_EN),
    'dolphin nebula':          ('SH2-308',_EN),
    'running man nebula':      ('SH2-279',_EN),

    # Reflection nebulae
    'pleiades':                ('M45',    _RN),
    'seven sisters':           ('M45',    _RN),
    'flaming star nebula':     ('IC405',  _RN),
    'flaming star':            ('IC405',  _RN),
    'iris nebula':             ('NGC7023',_RN),
    'witch head nebula':       ('IC2118', _RN),

    # Star fields / open clusters
    'double cluster':          ('NGC869', _SF),
    'beehive cluster':         ('M44',    _SF),
    'praesepe':                ('M44',    _SF),
    'butterfly cluster':       ('M6',     _SF),
    'ptolemy cluster':         ('M7',     _SF),

    # Wide field
    'sagittarius star cloud':  ('M24',    _WF),
    'milky way':               (None,     _WF),
    'large magellanic cloud':  (None,     _WF),
    'small magellanic cloud':  ('NGC292', _WF),
}


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_DATE_SUFFIX_RE = re.compile(
    r'[_\s-]\d{4}[-_]\d{2}[-_]\d{2}([_\s-]\d{2}[-_:]\d{2}([-_:]\d{2})?)?.*$'
)

_CATALOGUE_ID_RE = re.compile(
    r'\b('
    r'M\s*\d{1,3}'
    r'|NGC\s*\d{1,4}'
    r'|IC\s*\d{1,4}'
    r'|SH\s*2\s*[-_]\s*\d{1,3}'
    r'|SH2[-_]\d{1,3}'
    r'|LBN\s*\d{1,4}'
    r'|LDN\s*\d{1,4}'
    r'|B\s*\d{1,3}'
    r'|ARP\s*\d{1,3}'
    r')\b',
    re.IGNORECASE,
)

# FITS OBJECT values that mean "no target" — skip these
_EMPTY_OBJECT_VALUES = frozenset({
    '', 'unknown', 'none', 'target', 'object', 'n/a', 'na',
    'light', 'lights', 'flat', 'dark', 'bias', 'test',
})


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _normalise_id(raw: str) -> str:
    """Collapse whitespace and uppercase a raw catalogue ID token.

    Examples: 'M 31' → 'M31', 'NGC  7000' → 'NGC7000', 'Sh2-132' → 'SH2-132'.
    """
    s = raw.strip().upper()
    s = re.sub(r'^M\s+(\d)', r'M\1', s)
    s = re.sub(r'^(NGC|IC|LBN|LDN|ARP)\s+', r'\1', s)
    s = re.sub(r'^(B)\s+(\d)', r'\1\2', s)
    # Normalise Sharpless: SH 2-132, Sh2 132, sh2_132 → SH2-132
    s = re.sub(r'^SH\s*2\s*[-_\s]\s*(\d)', r'SH2-\1', s)
    return s


def _strip_date_suffix(name: str) -> str:
    """Remove a trailing date/time stamp from a folder name."""
    return _DATE_SUFFIX_RE.sub('', name).strip('_- ')


def _folder_to_label(directory: str) -> str:
    """Convert a directory path to a human-readable candidate name."""
    folder = os.path.basename(os.path.normpath(directory))
    stripped = _strip_date_suffix(folder)
    return stripped.replace('_', ' ').replace('-', ' ').strip()


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def _lookup_by_id(raw_id: str) -> Optional[Tuple[str, str, float]]:
    """Look up a catalogue ID in the static table.

    Returns (canonical_name, object_type, confidence) or None.
    """
    norm = _normalise_id(raw_id)
    entry = _CATALOGUE.get(norm)
    if entry:
        return entry[0], entry[1], 0.95
    return None


def _lookup_by_name(name: str) -> Optional[Tuple[str, str, float]]:
    """Look up a common name in the static table.

    Tries exact match first, then substring containment.
    Returns (canonical_name, object_type, confidence) or None.
    """
    key = name.lower().strip()

    # Exact match
    entry = _COMMON_NAMES.get(key)
    if entry:
        cat_id, obj_type = entry
        canonical = _CATALOGUE[cat_id][0] if cat_id and cat_id in _CATALOGUE else name
        return canonical, obj_type, 0.90

    # Substring: a known common name is wholly contained in the candidate, or vice versa
    for common_key, (cat_id, obj_type) in _COMMON_NAMES.items():
        if len(common_key) < 5:
            continue
        if common_key in key or key in common_key:
            canonical = _CATALOGUE[cat_id][0] if cat_id and cat_id in _CATALOGUE else common_key.title()
            return canonical, obj_type, 0.75

    return None


def _lookup_text(text: str) -> Optional[Tuple[str, str, float]]:
    """Search text for catalogue IDs or common names.

    Returns the highest-confidence match found, or None.
    """
    best: Optional[Tuple[str, str, float]] = None

    # 1. Catalogue ID regex — strongest signal
    for match in _CATALOGUE_ID_RE.finditer(text):
        result = _lookup_by_id(match.group(0))
        if result and (best is None or result[2] > best[2]):
            best = result

    # 2. Common name lookup on the whole string
    result = _lookup_by_name(text)
    if result and (best is None or result[2] > best[2]):
        best = result

    return best


# ---------------------------------------------------------------------------
# FITS header extraction
# ---------------------------------------------------------------------------

def _object_from_headers(frames: list) -> Optional[str]:
    """Return the most common non-empty FITS OBJECT keyword value."""
    values = []
    for f in frames:
        val = str(f.header.get('OBJECT', '')).strip()
        if val.lower() not in _EMPTY_OBJECT_VALUES:
            values.append(val)
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Simbad fallback
# ---------------------------------------------------------------------------

def _simbad_type_to_internal(otype: str) -> str:
    """Map a Simbad OTYPE string to an internal object_type key."""
    t = otype.lower()
    # Galaxy family
    if any(x in t for x in ('galaxy', 'gal', 'agn', 'seyfert', 'qso', 'liner', 'bcd')):
        return _G
    # Globular cluster
    if any(x in t for x in ('globular', 'glc', 'gl?', 'gc ')):
        return _GL
    # Planetary nebula
    if any(x in t for x in ('planetary', ' pn', 'pne', 'p.n.')):
        return _PN
    # Emission / HII / SNR
    if any(x in t for x in ('hii', 'h ii', 'emission', 'snr', 'supernova remnant',
                             'interstellar', 'ism', 'molecular')):
        return _EN
    # Reflection
    if 'reflection' in t or ' rne' in t:
        return _RN
    # Open cluster / stellar association
    if any(x in t for x in ('open', 'ocl', 'cl*', 'stellar association', 'opc')):
        return _SF
    return 'unknown'


def _simbad_lookup(name: str) -> Optional[Tuple[str, str, float]]:
    """Query Simbad for *name*; returns (canonical_name, object_type, 0.85) or None."""
    try:
        from astroquery.simbad import Simbad
        simbad = Simbad()
        simbad.add_votable_fields('otype', 'ids')
        result = simbad.query_object(name)
        if result is None or len(result) == 0:
            return None
        otype = str(result['OTYPE'][0])
        main_id = str(result['MAIN_ID'][0]).strip()
        obj_type = _simbad_type_to_internal(otype)
        if obj_type == 'unknown':
            return None
        return main_id, obj_type, 0.85
    except Exception as exc:
        _log.debug("Simbad lookup failed for %r: %s", name, exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def infer_target_from_metadata(
    directory: str,
    frames: list,
    use_simbad: bool = True,
) -> Tuple[Optional[str], Optional[str], float, Optional[str]]:
    """Infer target name and type from available metadata.

    Checks (in priority order):
      1. FITS OBJECT header keyword across accepted frames.
      2. Input directory name (date/time suffix stripped).
      3. Individual file name stems.
      4. Simbad query (if *use_simbad* is True and astroquery is installed).

    Returns
    -------
    target_name : str or None
    object_type : str or None   — key matching auto_settings TARGET_LABELS
    confidence  : float         — 0.0–1.0
    source      : str or None   — 'header' | 'folder' | 'filename' | 'simbad'
    """
    candidates: List[Tuple[str, str, float, str]] = []  # (name, type, conf, source)

    # ---- Source 1: FITS OBJECT header ----
    header_name = _object_from_headers(frames)
    if header_name:
        result = _lookup_text(header_name)
        if result:
            candidates.append((result[0], result[1], result[2], 'header'))
        elif use_simbad:
            simbad_result = _simbad_lookup(header_name)
            if simbad_result:
                candidates.append((simbad_result[0], simbad_result[1],
                                    simbad_result[2], 'simbad'))
            else:
                # Keep the raw header name at low confidence; type unknown
                candidates.append((header_name, 'unknown', 0.40, 'header'))

    # ---- Source 2: Folder name ----
    folder_label = _folder_to_label(directory)
    if folder_label:
        result = _lookup_text(folder_label)
        if result:
            # Slightly lower than a direct header hit; still high for a catalogue ID match
            folder_conf = result[2] * 0.95
            candidates.append((result[0], result[1], folder_conf, 'folder'))
        elif use_simbad and folder_label.lower() not in _EMPTY_OBJECT_VALUES:
            simbad_result = _simbad_lookup(folder_label)
            if simbad_result:
                candidates.append((simbad_result[0], simbad_result[1],
                                    simbad_result[2] * 0.90, 'simbad'))

    # ---- Source 3: Individual file name stems ----
    seen_stems: set = set()
    for f in frames[:20]:  # sample first 20 frames
        stem = os.path.splitext(os.path.basename(f.path))[0]
        # Only try stems that look like catalogue IDs (e.g. "NGC7000_001")
        id_match = _CATALOGUE_ID_RE.search(stem)
        if id_match and stem not in seen_stems:
            seen_stems.add(stem)
            result = _lookup_by_id(id_match.group(0))
            if result:
                candidates.append((result[0], result[1], result[2] * 0.80, 'filename'))

    if not candidates:
        return None, None, 0.0, None

    # Discard 'unknown' type candidates unless nothing better exists
    typed = [c for c in candidates if c[1] != 'unknown']
    pool = typed if typed else candidates

    # Pick highest confidence; break ties by source priority (header > folder > simbad > filename)
    source_priority = {'header': 4, 'folder': 3, 'simbad': 2, 'filename': 1}
    best = max(pool, key=lambda c: (c[2], source_priority.get(c[3], 0)))

    target_name, object_type, confidence, source = best
    _log.debug(
        "Target inference: %r → type=%s conf=%.2f source=%s",
        target_name, object_type, confidence, source,
    )
    return target_name, object_type, confidence, source
