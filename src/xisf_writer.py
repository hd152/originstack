"""Minimal PixInsight XISF 1.0 writer.

Writes a (H, W, 3) float32 RGB image as a valid XISF file that PixInsight,
Siril, and other XISF-aware tools can open.

XISF layout:
  Bytes 0-7:   signature  "XISF0100"
  Bytes 8-11:  uint32 LE  length of XML header block (including the 16-byte preamble up to here)
  Bytes 12-15: uint32 LE  reserved (0)
  Bytes 16-N:  XML header (UTF-8, no BOM)
  Bytes N-4095: zero padding to 4096-byte boundary
  Bytes 4096+: raw image data (planar float32, (C, H, W))

Reference: PixInsight XISF 1.0 Specification §7
"""
from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from typing import Dict, Optional

import numpy as np


def write_xisf(img: np.ndarray, output_path: str,
               header_meta: Optional[Dict[str, str]] = None) -> None:
    """Write a (H, W, 3) float32 RGB image to an XISF file.

    Args:
        img: Image array with shape (H, W, 3), any float dtype.
        output_path: Destination file path (should end in .xisf).
        header_meta: Optional dict of FITS-style keyword→value pairs to embed
                     as FITSKeyword elements in the XISF header.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) array, got shape {img.shape}")

    H, W, _ = img.shape
    # Store as planar (C, H, W) float32, row-major
    data = np.ascontiguousarray(
        np.transpose(img.astype(np.float32), (2, 0, 1))
    )
    data_bytes = data.tobytes()
    data_len = len(data_bytes)

    # Data starts at byte 4096 (fixed attachment offset)
    data_offset = 4096

    # Build XML header
    root = ET.Element("xisf", {
        "version": "1.0",
        "xmlns": "http://www.pixinsight.com/xisf",
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:schemaLocation":
            "http://www.pixinsight.com/xisf "
            "http://pixinsight.com/xisf/xisf-1.0.xsd",
    })

    ET.SubElement(root, "Metadata")

    image_el = ET.SubElement(root, "Image", {
        "type": "Float32",
        "geometry": f"{W}:{H}:3",
        "sampleFormat": "Float32",
        "colorSpace": "RGB",
        "location": f"attachment:{data_offset}:{data_len}",
    })

    if header_meta:
        fits_kw_el = ET.SubElement(image_el, "FITSKeywords")
        for key, val in header_meta.items():
            if val is None:
                continue
            ET.SubElement(fits_kw_el, "Keyword", {
                "name": str(key)[:8].upper(),
                "value": str(val),
                "comment": "",
            })

    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)

    # Build 16-byte XISF preamble
    # signature (8) + header_length uint32 LE (4) + reserved uint32 LE (4)
    header_length = len(xml_bytes)
    preamble = b"XISF0100" + struct.pack("<II", header_length, 0)

    # Pad entire header block to 4096 bytes
    header_block = preamble + xml_bytes
    if len(header_block) > data_offset:
        # Shouldn't happen for typical metadata sizes; extend block if needed
        data_offset_actual = ((len(header_block) + 4095) // 4096) * 4096
        # Rewrite location attribute with corrected offset
        image_el.set("location", f"attachment:{data_offset_actual}:{data_len}")
        xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        header_length = len(xml_bytes)
        preamble = b"XISF0100" + struct.pack("<II", header_length, 0)
        header_block = preamble + xml_bytes
        pad_to = data_offset_actual
    else:
        pad_to = data_offset

    header_block = header_block.ljust(pad_to, b"\x00")

    with open(output_path, "wb") as fh:
        fh.write(header_block)
        fh.write(data_bytes)
