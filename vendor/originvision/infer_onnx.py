"""Run an exported ONNX model (see export_onnx.py) on a single image and
print whatever predictions its heads support - the torch-free counterpart
of infer.py.

Needs only onnxruntime + opencv + numpy (requirements-infer.txt): no torch,
no astropy. FITS input is therefore out of scope here - debayering lives in
data/cache.py behind astropy. Feed a stacked master TIFF/PNG/JPG, or use
infer.py for raw .fits frames.

  python infer_onnx.py --model checkpoints/model.onnx --image FinalStackedMaster.tiff
"""

import argparse
import json
import os

import cv2
import numpy as np
import onnxruntime as ort

from data.imageops import apply_background_correction, resize_center_crop, stretch_to_uint8
from data.shape_features import gate_comet_prediction

FITS_EXTS = {".fits", ".fit", ".fts"}
# the comet shape-gate's thresholds were tuned against features computed on
# a 512px crop (data/shape_features.py) - independent of --size, which is
# whatever the model itself runs at and is often much smaller (256 default)
SHAPE_GATE_SIZE = 512


def load_any_image(path, size):
    ext = os.path.splitext(path)[1].lower()
    if ext in FITS_EXTS:
        raise SystemExit(
            f"{path}: infer_onnx.py does not handle FITS (needs astropy debayering). "
            "Use infer.py for raw .fits frames, or pass a stacked master TIFF/PNG."
        )
    rgb16 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if rgb16 is None:
        raise ValueError(f"could not read {path}")
    if rgb16.ndim == 3:
        rgb16 = cv2.cvtColor(rgb16, cv2.COLOR_BGR2RGB)
    img = stretch_to_uint8(rgb16)
    # shape-gate features, in BGR (cv2's native order, matching the
    # thresholds' tuning), independent of the model input's own --size
    feature_img = cv2.cvtColor(resize_center_crop(img, SHAPE_GATE_SIZE), cv2.COLOR_RGB2BGR)
    img = resize_center_crop(img, size)
    arr = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    return arr[np.newaxis], feature_img  # (1, 3, H, W)


def d4_views(arr):
    """The 8 flip/rotation views of a (1,C,H,W) array - every task target is
    invariant to them, so averaging outputs over all 8 cuts inference
    variance for free."""
    views = []
    for flip in (False, True):
        base = arr[:, :, :, ::-1] if flip else arr
        views.extend(np.rot90(base, k, axes=(2, 3)) for k in range(4))
    return [np.ascontiguousarray(v) for v in views]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def _meta(session):
    """ONNX custom metadata written by export_onnx.py."""
    m = session.get_modelmeta().custom_metadata_map
    epoch = m.get("epoch")
    if epoch is not None and epoch.lstrip("-").isdigit():
        epoch = int(epoch)  # infer.py reports it as an int; match that
    return {
        "tasks": m.get("tasks", "").split(",") if m.get("tasks") else [],
        "head_order": m.get("head_order", "").split(",") if m.get("head_order") else [],
        "categories": m.get("categories", "").split(",") if m.get("categories") else [],
        "exposures": [float(x) for x in m.get("exposures", "").split(",") if x],
        "quality_scale": float(m.get("quality_scale", "400.0")),
        "stray_light_threshold": float(m.get("stray_light_threshold", "27.0")),
        "epoch": epoch,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="checkpoints/model.onnx")
    parser.add_argument("--image", required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument(
        "--tta", action="store_true", help="average predictions over the 8 D4 flip/rotate views"
    )
    parser.add_argument(
        "--no-shape-gate",
        action="store_true",
        help="disable the comet shape gate (data/shape_features.py) - report the raw category "
        "prediction even when the shape features disagree",
    )
    parser.add_argument(
        "--extract-background",
        metavar="PATH",
        default=None,
        help="save a background-flattened PNG to PATH, using the background_grid head's "
        "prediction upsampled and subtracted (see README's background-extraction section)",
    )
    parser.add_argument(
        "--json", action="store_true", help="print a single-line JSON result instead of text"
    )
    args = parser.parse_args()

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    meta = _meta(sess)
    tasks = meta["tasks"] or meta["head_order"]

    imgs, feature_img = load_any_image(args.image, args.size)
    batches = d4_views(imgs) if args.tta else [imgs]
    raw_per_batch = [sess.run(None, {"image": b}) for b in batches]
    # background_grid is the one orientation-DEPENDENT output (a spatial
    # map, not a D4-invariant scalar/class) - naively averaging it across
    # rotated/flipped views would blend misaligned spatial predictions, so
    # it's excluded from the TTA average and just uses the first
    # (un-transformed) view.
    bg_idx = meta["head_order"].index("background_grid") if "background_grid" in tasks else None
    acc = [
        raw_per_batch[0][i]
        if i == bg_idx
        else sum(rb[i] for rb in raw_per_batch) / len(raw_per_batch)
        for i in range(len(raw_per_batch[0]))
    ]
    out = dict(zip(meta["head_order"], acc, strict=True))

    result = {"image": args.image, "checkpoint_epoch": meta["epoch"], "tasks": tasks}
    if "reject" in tasks:
        p = float(_sigmoid(out["reject"])[0])
        result["defect_probability"] = p
        result["is_defective"] = p > 0.5
    if "quality" in tasks:
        result["quality_score"] = float(out["quality"][0]) * meta["quality_scale"]
    if "category" in tasks:
        probs = _softmax(out["category"][0])
        cats = meta["categories"]
        order = np.argsort(probs)[::-1]
        top = int(order[0])
        picked = top if args.no_shape_gate else gate_comet_prediction(probs, feature_img, cats)
        result["category"] = cats[picked]
        result["category_confidence"] = float(probs[picked])
        result["top_categories"] = [[cats[i], float(probs[i])] for i in order[:3]]
        result["category_shape_gated"] = picked != top
    if "exposure" in tasks:
        probs = _softmax(out["exposure"][0])
        i = int(np.argmax(probs))
        result["predicted_exposure_s"] = meta["exposures"][i]
        result["exposure_confidence"] = float(probs[i])
    if "sky_brightness" in tasks:
        result["sky_brightness"] = float(out["sky_brightness"][0]) * 255
    if "stray_light_gradient" in tasks:
        g = float(out["stray_light_gradient"][0]) * 255
        result["stray_light_gradient"] = g
        result["stray_light_flag"] = g > meta["stray_light_threshold"]
    if "trailing" in tasks:
        t = float(out["trailing"][0])
        result["trailing_score"] = t
        result["trailing_flag"] = t > 0.35
    if "background_grid" in tasks and args.extract_background:
        # applied to the 512px shape-gate crop (same relative framing as the
        # model's own input, higher resolution) rather than the --size array
        corrected = apply_background_correction(feature_img, out["background_grid"][0])
        cv2.imwrite(args.extract_background, corrected)
        result["background_extracted_to"] = args.extract_background

    if args.json:
        print(json.dumps(result))
        return

    print(f"{result['image']}  (checkpoint epoch {result['checkpoint_epoch']}, tasks={tasks})")
    if "defect_probability" in result:
        print(
            f"  defect probability : {result['defect_probability']:.3f}  ({'LIKELY DEFECTIVE' if result['is_defective'] else 'looks OK'})"
        )
    if "quality_score" in result:
        print(f"  predicted quality   : {result['quality_score']:.1f}")
    if "category" in result:
        gated_note = (
            "  (raw CNN top pick was comet, shape features overrode it)"
            if result["category_shape_gated"]
            else ""
        )
        print(
            f"  predicted category  : {result['category']} ({result['category_confidence']:.2f})"
            f"{gated_note}"
        )
        print(
            "  top categories      :",
            ", ".join(f"{n} ({p:.2f})" for n, p in result["top_categories"]),
        )
    if "predicted_exposure_s" in result:
        print(
            f"  predicted exposure  : {result['predicted_exposure_s']:.0f}s "
            f"({result['exposure_confidence']:.2f})"
        )
    if "sky_brightness" in result:
        print(
            f"  sky brightness      : {result['sky_brightness']:.1f} / 255 (image-only proxy, not true Bortle scale)"
        )
    if "stray_light_gradient" in result:
        flag = "POSSIBLE STRAY LIGHT" if result["stray_light_flag"] else "even background"
        print(f"  stray light gradient: {result['stray_light_gradient']:.1f} / 255  ({flag})")
    if "trailing_score" in result:
        flag = "STAR TRAILING" if result["trailing_flag"] else "round stars"
        print(f"  trailing score      : {result['trailing_score']:.2f} / 1  ({flag})")
    if "background_extracted_to" in result:
        print(f"  background-flattened image written to: {result['background_extracted_to']}")


if __name__ == "__main__":
    main()
