"""--export png / png16: lossless copies of the preview, from the same render as the JPEG."""
import argparse
import os
import struct
import zlib

import numpy as np
import pytest
from PIL import Image

from src.io_fits import (
    export_paths,
    preview_exports,
    render_preview_rgb_float,
    save_preview_rgb,
    to_uint8,
    write_final_images,
    write_png16,
)


def _read_png16(path):
    """Decode our own 16-bit RGB PNG (IHDR, one IDAT, Sub-filtered rows) independently."""
    data = open(path, "rb").read()
    assert data[:8] == bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
    pos, chunks = 8, {}
    while pos < len(data):
        n = struct.unpack(">I", data[pos:pos + 4])[0]
        kind, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + n]
        assert struct.unpack(">I", data[pos + 8 + n:pos + 12 + n])[0] == zlib.crc32(kind + body) & 0xFFFFFFFF
        chunks.setdefault(kind, b"")
        chunks[kind] += body
        pos += 12 + n
    w, h, depth, ctype = struct.unpack(">IIBB", chunks[b"IHDR"][:10])
    assert (depth, ctype) == (16, 2)
    rows = np.frombuffer(zlib.decompress(chunks[b"IDAT"]), np.uint8).reshape(h, 1 + w * 6)
    assert np.all(rows[:, 0] == 1)
    raw = rows[:, 1:].astype(np.int64)
    for x in range(6, raw.shape[1]):                     # undo 'Sub': byte + the one 6 earlier
        raw[:, x] = (raw[:, x] + raw[:, x - 6]) & 0xFF
    return raw.astype(np.uint8).view(">u2").reshape(h, w, 3).astype(np.uint16)


def _img(h=37, w=53):
    rng = np.random.default_rng(0)
    return rng.uniform(0, 1, (h, w, 3)).astype(np.float32)


def test_png16_round_trips_exactly(tmp_path):
    img = _img()
    p = str(tmp_path / "x.png")
    write_png16(p, img)
    got = _read_png16(p)
    np.testing.assert_array_equal(got, np.rint(img * 65535).astype(np.uint16))
    with Image.open(p) as im:                             # a valid PNG for other readers too
        im.verify()
    with Image.open(p) as im:
        assert im.size == (53, 37)


def test_final_images_share_one_render(tmp_path):
    img = _img()
    jpg = str(tmp_path / "stack.jpg")
    write_final_images(img, jpg, ("png", "png16"))
    paths = export_paths(jpg)
    assert paths == {"png": str(tmp_path / "stack.png"), "png16": str(tmp_path / "stack_16bit.png")}
    np.testing.assert_array_equal(np.asarray(Image.open(paths["png"])), to_uint8(img))   # lossless
    np.testing.assert_array_equal(_read_png16(paths["png16"]), np.rint(img * 65535).astype(np.uint16))
    assert os.path.exists(jpg) and not any(f.endswith(".tmp") for f in os.listdir(tmp_path))


def test_no_exports_writes_only_the_jpeg(tmp_path):
    jpg = str(tmp_path / "stack.jpg")
    write_final_images(_img(), jpg)
    assert os.listdir(tmp_path) == ["stack.jpg"]


def test_cli_export_flags():
    from src.cli import parse_args
    a = parse_args(["-d", "x", "-o", "y.fits", "--export", "tiff,png,png16"])
    assert a.output_tiff and a.output_png and a.output_png16
    assert preview_exports(a) == ("png", "png16")
    assert preview_exports(parse_args(["-d", "x", "-o", "y.fits"])) == ()
    with pytest.raises(SystemExit):
        parse_args(["-d", "x", "-o", "y.fits", "--export", "webp"])


def test_restretch_rewrites_the_exports(tmp_path):
    from src.ui_events import UIEvents
    rng = np.random.default_rng(1)
    rgb = (100.0 + rng.normal(0, 5, (64, 80, 3))).astype(np.float32)
    rgb[20:40, 30:50] += 400.0
    jpg = str(tmp_path / "stack.jpg")
    save_preview_rgb(rgb, jpg, stretch="ghs", exports=("png", "png16"))
    paths = export_paths(jpg)
    before = open(paths["png"], "rb").read()
    ui = UIEvents()
    ui.attach()
    ui.remember_final_preview(rgb, jpg, argparse.Namespace(stretch="ghs", stretch_color="preserve",
                                                          output_png=True, output_png16=True))
    ui.restretch("linear", "channel")
    assert open(paths["png"], "rb").read() != before
    ref = render_preview_rgb_float(rgb, stretch="linear", color="channel")
    np.testing.assert_array_equal(np.asarray(Image.open(paths["png"])), to_uint8(ref))
    np.testing.assert_array_equal(_read_png16(paths["png16"]), np.rint(ref * 65535).astype(np.uint16))


def test_app_checkboxes_join_the_typed_export_formats():
    from src.desktop_app import with_export_checkboxes
    from src.desktop_control import build_argv_from_form
    assert with_export_checkboxes({"directory": "d"}, "", {"png": False, "png16": False}) == {"directory": "d"}
    f = with_export_checkboxes({"directory": "d"}, "", {"png": True, "png16": True})
    assert f["export"] == "png,png16"
    f = with_export_checkboxes({"export": "tiff, png"}, "tiff, png", {"png": True, "png16": True})
    assert f["export"] == "tiff,png,png16"                      # typed first, each once
    argv = build_argv_from_form({"directory": "d", "output": "o.fits", "export": "tiff,png16"})
    assert argv[argv.index("--export") + 1] == "tiff,png16"
