"""Generate OriginStack's icon and logo.

The artwork is produced from code rather than committed as an opaque binary,
so it can be re-rendered at any size, tweaked in one place, and diffed
meaningfully -- the same preference for generated-over-vendored that put
``src/wavelet.py`` and ``src/net_query.py`` in this codebase.

    python tools/make_icon.py

Writes:
    packaging/icon.ico   multi-resolution app icon (16-256 px)
    assets/icon.png      the mark alone, 512 px, transparent
    assets/logo.png      mark + wordmark for light pages
    assets/logo-dark.png same, for dark pages

Design: three stacked frames converging upward into a star. The stack is the
literal operation this tool performs -- many exposures folded into one -- and
the star is what comes out of it. Everything is built from bold silhouettes
because the icon has to survive a 16x16 taskbar, where fine detail turns to
mud; the ASCII preview at the bottom of this file exists to check exactly
that, at the size where it actually matters.

Only Pillow is needed, which the project already depends on for previews.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent

# Rendered large and downsampled: Pillow has no anti-aliased polygon fill, so
# supersampling is what buys clean diagonals on the stacked frames.
SUPERSAMPLE = 8
BASE = 256

# Deep-sky palette: indigo field, cyan-white light.
BG_TOP = (28, 24, 62)
BG_BOTTOM = (12, 10, 30)
FRAME_FAR = (86, 104, 190)
FRAME_MID = (128, 158, 228)
FRAME_NEAR = (198, 226, 255)
STAR = (226, 245, 255)
GLOW = (96, 176, 255)


def _vertical_gradient(size: int, top: tuple, bottom: tuple) -> Image.Image:
    grad = Image.new('RGB', (1, size))
    px = grad.load()
    for y in range(size):
        t = y / max(size - 1, 1)
        px[0, y] = tuple(int(a + (b - a) * t) for a, b in zip(top, bottom))
    return grad.resize((size, size), Image.BILINEAR)


def _rounded_mask(size: int, radius_frac: float = 0.22) -> Image.Image:
    mask = Image.new('L', (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, size - 1, size - 1], radius=int(size * radius_frac), fill=255)
    return mask


def _four_point_star(draw: ImageDraw.ImageDraw, cx: float, cy: float,
                     r: float, waist: float, colour: tuple) -> None:
    """A four-point sparkle: long axes, pinched waist. Reads at any size."""
    w = r * waist
    draw.polygon([(cx, cy - r), (cx + w, cy - w), (cx + r, cy),
                  (cx + w, cy + w), (cx, cy + r), (cx - w, cy + w),
                  (cx - r, cy), (cx - w, cy - w)], fill=colour)


def render_mark(size: int = BASE, background: bool = True) -> Image.Image:
    """The icon mark: three stacked frames beneath a star."""
    s = size * SUPERSAMPLE
    img = Image.new('RGBA', (s, s), (0, 0, 0, 0))

    if background:
        bg = _vertical_gradient(s, BG_TOP, BG_BOTTOM).convert('RGBA')
        bg.putalpha(_rounded_mask(s))
        img.alpha_composite(bg)

    draw = ImageDraw.Draw(img)

    # --- Three stacked frames, in perspective -----------------------------
    # Each higher plate is narrower, so the stack reads as receding upward
    # and converging on the star.
    cx = s * 0.5
    plates = [
        (s * 0.775, s * 0.305, s * 0.050, FRAME_NEAR),   # y, half-width, height
        (s * 0.676, s * 0.248, s * 0.044, FRAME_MID),
        (s * 0.588, s * 0.191, s * 0.038, FRAME_FAR),
    ]
    for cy, half, h, colour in plates:
        # A shallow parallelogram: top edge inset from the bottom edge gives
        # the plate a sense of tilt without needing a real 3D projection.
        inset = half * 0.18
        draw.polygon([
            (cx - half, cy),
            (cx + half, cy),
            (cx + half - inset, cy - h),
            (cx - half + inset, cy - h),
        ], fill=colour + (255,))

    # --- The star the stack produces --------------------------------------
    star_cx, star_cy = cx, s * 0.375
    glow = Image.new('RGBA', (s, s), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse(
        [star_cx - s * 0.145, star_cy - s * 0.145,
         star_cx + s * 0.145, star_cy + s * 0.145],
        fill=GLOW + (120,))
    glow = glow.filter(ImageFilter.GaussianBlur(s * 0.045))
    img.alpha_composite(glow)

    draw = ImageDraw.Draw(img)
    _four_point_star(draw, star_cx, star_cy, s * 0.175, 0.17, STAR + (255,))

    if background:
        out = Image.new('RGBA', (s, s), (0, 0, 0, 0))
        out.paste(img, (0, 0), _rounded_mask(s))
        img = out

    return img.resize((size, size), Image.LANCZOS)


def _load_font(px: int):
    for name in ('segoeuib.ttf', 'arialbd.ttf', 'DejaVuSans-Bold.ttf'):
        try:
            return ImageFont.truetype(name, px)
        except Exception:
            continue
    return ImageFont.load_default()


def render_logo(height: int = 256, theme: str = 'light') -> Image.Image:
    """Mark plus wordmark, on transparency, for the README header.

    Two variants exist because the wordmark sits on transparency and a README
    is rendered on both a white and a near-black page. A single near-white
    wordmark vanishes on the light one; a single dark wordmark vanishes on the
    dark one. ``theme`` names the *page* the logo will sit on, and the README
    selects between them with a ``<picture>`` + ``prefers-color-scheme``.
    """
    if theme == 'dark':
        name_fill, tag_fill = (238, 244, 255, 255), (150, 172, 215, 255)
    else:
        name_fill, tag_fill = (26, 30, 62, 255), (92, 108, 155, 255)

    mark = render_mark(height)
    font = _load_font(int(height * 0.40))
    sub_font = _load_font(int(height * 0.145))

    pad = int(height * 0.28)
    probe = ImageDraw.Draw(Image.new('RGBA', (1, 1)))
    name_w = int(probe.textlength('OriginStack', font=font))
    tag_w = int(probe.textlength('astrophotography stacking', font=sub_font))
    width = height + pad + max(name_w, tag_w) + pad // 2

    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    img.alpha_composite(mark, (0, 0))

    draw = ImageDraw.Draw(img)
    x = height + pad
    draw.text((x, int(height * 0.22)), 'OriginStack', font=font, fill=name_fill)
    draw.text((x + 2, int(height * 0.685)), 'astrophotography stacking',
              font=sub_font, fill=tag_fill)
    return img


def ascii_preview(img: Image.Image, size: int = 16) -> str:
    """Check the silhouette survives a taskbar. This is the real test."""
    small = img.convert('RGBA').resize((size, size), Image.LANCZOS)
    px = small.load()
    ramp = ' .:-=+*#%@'
    rows = []
    for y in range(size):
        row = ''
        for x in range(size):
            r, g, b, a = px[x, y]
            lum = (0.299 * r + 0.587 * g + 0.114 * b) * (a / 255.0)
            row += ramp[min(len(ramp) - 1, int(lum / 26))]
        rows.append(row)
    return '\n'.join(rows)


def main() -> int:
    assets = ROOT / 'assets'
    assets.mkdir(exist_ok=True)

    mark = render_mark(512)
    mark.save(assets / 'icon.png')

    render_logo(256, theme='light').save(assets / 'logo.png')
    render_logo(256, theme='dark').save(assets / 'logo-dark.png')

    ico_sizes = [16, 32, 48, 64, 128, 256]
    # Render each size independently rather than letting the .ico writer
    # downscale one bitmap: the small sizes need their own supersampled pass
    # or the thin plate edges alias away to nothing.
    frames = [render_mark(n) for n in ico_sizes]
    frames[-1].save(ROOT / 'packaging' / 'icon.ico', format='ICO',
                    sizes=[(n, n) for n in ico_sizes],
                    append_images=frames[:-1])

    print(f"wrote {assets / 'icon.png'}")
    print(f"wrote {assets / 'logo.png'} (for light pages)")
    print(f"wrote {assets / 'logo-dark.png'} (for dark pages)")
    print(f"wrote {ROOT / 'packaging' / 'icon.ico'} ({', '.join(map(str, ico_sizes))} px)")
    print('\n16x16 silhouette check:')
    print(ascii_preview(mark))
    return 0


if __name__ == '__main__':
    sys.exit(main())
