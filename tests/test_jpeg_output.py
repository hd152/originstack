"""Preview JPEGs: rounded to 8 bits, colour at full resolution (4:4:4)."""
import io

import numpy as np
from PIL import Image, JpegImagePlugin

from src.io_fits import JPEG_SAVE_KW, preview_jpeg_from_uint8, to_uint8, write_preview_jpeg


def test_to_uint8_rounds_instead_of_truncating():
    x = np.array([0.0, 0.4, 0.6, 1.4, 254.6, 300.0, -3.0], np.float32) / 255.0
    np.testing.assert_array_equal(to_uint8(x), [0, 0, 1, 1, 255, 255, 0])
    rng = np.random.default_rng(0)
    sky = rng.uniform(0, 3, 100000).astype(np.float32) / 255.0      # a dark preview sky
    assert abs(float(to_uint8(sky).mean()) - float(sky.mean() * 255)) < 0.01


def _red_stars():
    """Small coloured stars on a black sky, as in a stretched preview."""
    img = np.zeros((128, 128, 3), np.float32)
    yy, xx = np.mgrid[:128, :128]
    for y in range(16, 128, 32):
        for x in range(16, 128, 32):
            g = np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 1.0 ** 2))
            img += g[..., None] * np.array([1.0, 0.45, 0.2], np.float32)
    return to_uint8(np.clip(img, 0, 1))


def _star_chroma(u8):
    c = u8.astype(np.float64)[16::32, 16::32]
    return float(np.mean(c[..., 0] - c[..., 2]))


def test_file_jpeg_keeps_full_resolution_colour(tmp_path):
    u8 = _red_stars()
    path = str(tmp_path / "p.jpg")
    write_preview_jpeg(u8, path)
    with Image.open(path) as im:
        assert JpegImagePlugin.get_sampling(im) == 0                  # 4:4:4
        back = np.asarray(im.convert("RGB"))
    # 4:2:0 kept ~60% of these stars' colour; 4:4:4 keeps nearly all of it
    assert _star_chroma(back) > 0.9 * _star_chroma(u8)
    buf = io.BytesIO()
    Image.fromarray(u8).save(buf, format="JPEG", quality=95)          # Pillow's default
    assert _star_chroma(np.asarray(Image.open(io.BytesIO(buf.getvalue())))) < 0.8 * _star_chroma(u8)


def test_viewer_jpeg_keeps_full_resolution_colour():
    with Image.open(io.BytesIO(preview_jpeg_from_uint8(_red_stars()))) as im:
        assert JpegImagePlugin.get_sampling(im) == 0
    assert JPEG_SAVE_KW == {"subsampling": 0}
