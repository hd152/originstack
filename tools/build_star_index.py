"""Pre-fill the local Gaia star index used by ``--plate-solver local``/``auto``.

The solver downloads (and keeps) the index tiles a field needs the first time
that field is solved online. Run this ahead of time on a machine that will
work offline:

    python tools/build_star_index.py --all                  # whole sky, ~1650 tiles, ~80 MB
    python tools/build_star_index.py --ra 198.96 --dec 42.03 --radius 10
    python tools/build_star_index.py --dec-min -30           # everything visible from ~60N
    python tools/build_star_index.py --status

Tiles land in ``src.local_solve.star_index_dir()`` (override with
``--root`` or ``$ORIGINSTACK_STAR_INDEX``). Existing tiles are skipped unless
``--refresh``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.local_solve import (  # noqa: E402
    all_tiles,
    build_index,
    star_index_dir,
    tile_bounds,
    tiles_for_cone,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--all', action='store_true', help='every tile on the sky')
    ap.add_argument('--ra', type=float, help='cone centre RA (deg)')
    ap.add_argument('--dec', type=float, help='cone centre Dec (deg)')
    ap.add_argument('--radius', type=float, default=5.0, help='cone radius (deg)')
    ap.add_argument('--dec-min', type=float, default=None, help='all tiles north of this Dec')
    ap.add_argument('--dec-max', type=float, default=None, help='all tiles south of this Dec')
    ap.add_argument('--root', default=None, help=f'index directory (default {star_index_dir()})')
    ap.add_argument('--refresh', action='store_true', help='re-download existing tiles')
    ap.add_argument('--status', action='store_true', help='report coverage and exit')
    a = ap.parse_args(argv)
    root = a.root or star_index_dir()

    if a.status:
        tiles = all_tiles()
        have = [t for t in tiles
                if os.path.exists(os.path.join(root, f'b{t[0]:02d}_r{t[1]:03d}.npy'))]
        size = sum(os.path.getsize(os.path.join(root, f)) for f in os.listdir(root)) \
            if os.path.isdir(root) else 0
        print(f"{root}: {len(have)}/{len(tiles)} tiles, {size / 1e6:.1f} MB")
        return 0

    if a.all:
        tiles = all_tiles()
    elif a.ra is not None and a.dec is not None:
        tiles = tiles_for_cone(a.ra, a.dec, a.radius)
    elif a.dec_min is not None or a.dec_max is not None:
        lo = -90.0 if a.dec_min is None else a.dec_min
        hi = 90.0 if a.dec_max is None else a.dec_max
        tiles = [t for t in all_tiles() if tile_bounds(*t)[3] > lo and tile_bounds(*t)[2] < hi]
    else:
        ap.error('give --all, --ra/--dec[/--radius], --dec-min/--dec-max, or --status')

    t0 = time.time()

    def progress(k, n, ok, bad):
        el = time.time() - t0
        print(f"\r  {k}/{n} tiles  fetched {ok}  failed {bad}  {el:.0f}s", end='', flush=True)

    print(f"Index: {root} -- {len(tiles)} tile(s) requested")
    ok, bad = build_index(tiles, root=root, refresh=a.refresh, progress=progress)
    print(f"\nDone: {ok} fetched, {bad} failed, {len(tiles) - ok - bad} already present")
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
