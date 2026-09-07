"""Hand-engineered shape features that gate the category head's "comet"
prediction - cv2 + numpy only (no torch/astropy), so infer_onnx.py's slim
env can use it too. Not a general accuracy lever: a comet-vs-others
discriminator for one specific, diagnosed failure mode.

Why: the trained category head has comet precision ~0.33 - focal loss fixed
comet's recall (it was stuck at 0.11 for 25 epochs without it) but overshot,
and the head now treats "comet" as a catch-all for ambiguous frames rather
than a real detector. The two biggest false-positive sources, measured on
the real val set:

  - nebula frames called comet: the model is picking the brightest patch of
    an off-center diffuse structure. Real comets are a single tracked
    target and sit near the frame center; these false positives averaged
    center_dist_norm=0.81 (see below) vs a real comet's ~0.05.
  - star_cluster frames called comet: these still have a whole field of
    discrete stars in them (~107 bright components, same as correctly
    classified star_clusters), unlike a real comet's handful (~44-61).

Two features, both from the largest bright connected component in the
frame:
  - n_components: count of bright blobs above a percentile threshold - a
    comet has a handful (nucleus + tail + maybe a few field stars); a star
    field has dozens more.
  - center_dist_norm: the largest blob's centroid distance from the frame
    center, normalised by the half-diagonal (0 = dead center, ~1 = a
    corner) - comets are centered, the confused nebula patches usually
    aren't.

COMET_GATE_CENTER_DIST / COMET_GATE_N_COMPONENTS were grid-searched against
real category_acc_frames on the val set (not eyeballed): the search swept
center_dist in [0.15..0.6] x n_components in [40..120] and picked the
combo maximising overall accuracy. Result on that val set: comet precision
0.327 -> 0.820, recall 0.758 -> 0.758, category_acc_frames 0.8751 -> 0.8815.

Compute these on a reasonably high-resolution crop, not whatever --size the
CNN itself runs at - features measured on a heavily downsampled image lose
the fine structure (individual stars, a thin tail) they depend on. This
project computes them on a 512px center crop (matching data/cache.py's
default), independent of the model's own --size.
"""

import cv2
import numpy as np

# grid-searched on the val set - see module docstring
COMET_GATE_CENTER_DIST = 0.20
COMET_GATE_N_COMPONENTS = 80


def blob_shape_features(img, thresh_percentile=99.0, min_area=5):
    """Return (n_components, center_dist_norm) for the largest bright
    connected component in a BGR or grayscale uint8 image (BGR to match
    cv2.imread's native order and data/light_stats.py's convention - the
    thresholds above were tuned against images read that way). An image
    with no component above threshold returns (0, 1.0): "nothing found,
    definitely not centered".

    Known edge case: if the brightest ~(100-thresh_percentile)% of pixels
    are a single flat, saturated value (no internal variance), the
    percentile can land exactly on that value and cv2.threshold's strict
    `>` excludes it, returning nothing found. Real astro frames always have
    background texture/noise so this hasn't shown up in practice; noted for
    anyone feeding this a synthetic or otherwise unusually flat image (see
    tests/test_shape_features.py's synthetic-image helpers for how to avoid
    it)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape
    thresh_val = max(float(np.percentile(gray, thresh_percentile)), 1.0)
    _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    n_labels, _labels_im, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    comps = [i for i in range(1, n_labels) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if not comps:
        return 0, 1.0
    largest = max(comps, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    cx, cy = centroids[largest]
    center_dist = ((cx - w / 2) ** 2 + (cy - h / 2) ** 2) ** 0.5
    center_dist_norm = center_dist / (0.5 * (w**2 + h**2) ** 0.5)
    return len(comps), center_dist_norm


def gate_comet_prediction(probs, img, categories, comet_index=None):
    """probs: 1D array of per-class probabilities (softmax already applied,
    same order as `categories`). img: the BGR/grayscale uint8 image the
    prediction came from (see module docstring re: resolution). Returns the
    predicted class index - the raw argmax, unless it's "comet" and the
    shape features disagree, in which case falls back to the 2nd-best
    class. No-op whenever the top prediction isn't comet, so this only ever
    makes the comet head more conservative, never touches the other three
    classes' own predictions."""
    if comet_index is None:
        comet_index = list(categories).index("comet")
    order = np.argsort(probs)[::-1]
    top = int(order[0])
    if top != comet_index:
        return top
    n_components, center_dist_norm = blob_shape_features(img)
    if center_dist_norm > COMET_GATE_CENTER_DIST or n_components > COMET_GATE_N_COMPONENTS:
        return int(order[1])
    return top
