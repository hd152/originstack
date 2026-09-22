"""Robust PCA (Principal Component Pursuit) master calibration frames.

Decomposes a calibration stack (bias/dark/flat) into a low-rank component --
the true shared pattern (flat-field vignetting, fixed dark-current map) -- and
a sparse component (dust motes that shifted between sessions, transient hot
pixels, cosmic ray hits). Unlike a per-pixel median, which treats every
outlier independently, RPCA models the whole stack jointly: a dust donut that
moved is sparse *in the stack*, not just an outlier at one pixel, so the
decomposition isolates it explicitly rather than relying on order statistics
at each pixel to happen to reject it.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage

from src.models import Config, FrameInfo
from src.utils import get_logger, safe_print

try:
    import astro_native as _native
    _HAS_NATIVE = True
except Exception:
    _native = None
    _HAS_NATIVE = False

_log = get_logger()


def _thin_svd_wide(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thin SVD of a wide ``(N, P)`` matrix (``N << P``, the calibration-stack
    shape -- N frames, P flattened pixels) via the Gram-matrix trick:
    eigendecompose the small ``N x N`` Gram matrix ``M @ M.T`` instead of
    calling a general SVD directly on the full matrix. Falls back to
    ``np.linalg.svd`` when the matrix isn't wide (the trick's whole benefit
    assumes ``N << P``) or when N/P are degenerate.

    Measured on the realistic robust-PCA problem shape this exists for
    (N=20, P=18,000,000): direct ``np.linalg.svd`` 37.8s; the Gram trick in
    plain numpy 15.7s (~2.4x, numpy's own SVD routine isn't specialized for
    this shape); with the native ``gram_matrix_wide``/``small_times_wide``
    kernels (rayon-parallel, `ext/astro_native/src/lib.rs`) 7.9s (~4.8x on
    top of the algorithmic win, ~9x combined vs. the naive direct-SVD
    baseline) -- numpy's own ``@`` for this shape got zero benefit from this
    machine's cores (8.1s default-threaded vs 8.9s forced single-threaded),
    which is the headroom the native kernels close.

    Returns ``(U, s, Vt)`` with descending singular values, satisfying
    ``M ~= (U * s) @ Vt`` -- note the sign convention of each ``(u, v)``
    pair may differ from ``np.linalg.svd``'s (SVD signs are only defined up
    to a simultaneous flip); callers that only use the triple to reconstruct
    a matrix (as ``robust_pca_decompose`` does) are unaffected.
    """
    n, p = M.shape
    if p < n:
        return np.linalg.svd(M, full_matrices=False)

    M_c = np.ascontiguousarray(M, dtype=np.float64)
    G = None
    if _HAS_NATIVE:
        try:
            G = np.asarray(_native.gram_matrix_wide(M_c))
        except Exception:
            G = None
    if G is None:
        G = M_c @ M_c.T

    eigvals, eigvecs = np.linalg.eigh(G)  # ascending
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    U = eigvecs[:, order]
    s = np.sqrt(np.maximum(eigvals, 0.0))

    s_safe = np.where(s > 1e-300, s, 1.0)
    Vt = None
    if _HAS_NATIVE:
        try:
            Vt = np.asarray(_native.small_times_wide(
                np.ascontiguousarray(U.T), M_c)) / s_safe[:, None]
        except Exception:
            Vt = None
    if Vt is None:
        Vt = (U.T @ M_c) / s_safe[:, None]

    return U, s, Vt


def robust_pca_decompose(D: np.ndarray, max_iters: int = Config.ROBUST_PCA_MAX_ITERS,
                         tol: float = Config.ROBUST_PCA_TOL,
                         lam: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Split ``D`` (n_samples, n_features) into low-rank ``L`` + sparse ``S``.

    Inexact Augmented Lagrange Multiplier (IALM) solver for Principal
    Component Pursuit (Candes, Li, Ma & Wright 2011): minimise
    ``||L||_* + lam * ||S||_1`` subject to ``D = L + S``. Each iteration is
    one economy SVD (soft-thresholded singular values -> L) and one
    elementwise soft-threshold (-> S) -- the SVD is via ``_thin_svd_wide``'s
    Gram-matrix trick (see its docstring for the ~9x measured speedup over
    calling ``np.linalg.svd`` directly), which is what makes this practical
    at all for the small-N, huge-P calibration-stack shape.

    ``lam`` defaults to the standard ``1/sqrt(max(n_samples, n_features))``
    (Candes et al.'s recommended value; no dataset-specific tuning needed).
    """
    D = np.asarray(D, dtype=np.float64)
    n, p = D.shape
    if lam is None:
        lam = 1.0 / np.sqrt(max(n, p))

    # Thin SVD of D itself gives both the spectral norm (for the initial
    # step size mu) and re-uses the same economy decomposition shape the
    # per-iteration update below performs.
    _, sv0, _ = _thin_svd_wide(D)
    norm_two = float(sv0[0]) if sv0.size else 1.0
    norm_fro = float(np.linalg.norm(D, 'fro'))
    if norm_fro < 1e-30:
        return D.copy(), np.zeros_like(D)

    mu = 1.25 / max(norm_two, 1e-12)
    mu_bar = mu * 1e7
    rho = 1.5

    # S/Y allocated once and reused every iteration (native writes them in
    # place) instead of the ~dozen fresh (N,P) temporaries/iteration the
    # plain-numpy expressions below build -- found by profiling a real
    # --flat-from-lights run (N=10, P=6.25M): robust_pca_decompose was
    # spending 100s of a 126s total in exactly this elementwise arithmetic
    # around the SVD, not the SVD itself.
    L = np.zeros_like(D)
    S = np.zeros_like(D)
    Y = np.zeros_like(D)
    D_c = np.ascontiguousarray(D)

    for _ in range(max_iters):
        # L-update: singular value shrinkage (proximal operator of the nuclear norm)
        svd_input = None
        if _HAS_NATIVE:
            try:
                svd_input = np.asarray(_native.robust_pca_pre_svd_input(D_c, S, Y, mu))
            except Exception:
                svd_input = None
        if svd_input is None:
            svd_input = D - S + Y / mu
        U, sigma, Vt = _thin_svd_wide(svd_input)
        sigma_shrunk = np.maximum(sigma - 1.0 / mu, 0.0)
        # (U * sigma_shrunk) @ Vt is the exact same small-(N,N)-times-wide-
        # (N,P) shape _thin_svd_wide's own Gram-trick GEMM is (small_times_
        # wide above) -- routing it through the same native kernel instead of
        # numpy's plain `@` matters because this machine's numpy has no
        # optimized BLAS (numpy.show_config reports `blas: name: auto`,
        # measured ~1.4 GFLOPS, reference-BLAS-speed): a lone `np.matmul`
        # call here was 32.9s of this function's 62.5s once the elementwise
        # arithmetic around it was already fused (profiled after the fusion
        # above, not guessed).
        weighted_U = np.ascontiguousarray(U * sigma_shrunk)
        L = None
        if _HAS_NATIVE:
            try:
                L = np.asarray(_native.small_times_wide(weighted_U, np.ascontiguousarray(Vt)))
            except Exception:
                L = None
        if L is None:
            L = weighted_U @ Vt

        # S-update + residual + Y-update + residual norm (proximal operator
        # of the L1 norm, fused): native writes S and Y in place.
        lam_over_mu = lam / mu
        resid_norm = None
        if _HAS_NATIVE:
            try:
                resid_norm = float(_native.robust_pca_iterate(D_c, L, S, Y, lam_over_mu, mu))
            except Exception:
                resid_norm = None
        if resid_norm is None:
            temp = D - L + Y / mu
            S[...] = np.sign(temp) * np.maximum(np.abs(temp) - lam_over_mu, 0.0)
            residual = D - L - S
            Y += mu * residual
            resid_norm = float(np.linalg.norm(residual, 'fro'))

        mu = min(mu * rho, mu_bar)

        if resid_norm / norm_fro < tol:
            break

    return L, S


def _bayer_planes(img: np.ndarray) -> List[np.ndarray]:
    """The 4 same-colour sub-planes of a 2x2-CFA mosaic, R/G1/G2/B order."""
    return [img[0::2, 0::2], img[0::2, 1::2], img[1::2, 0::2], img[1::2, 1::2]]


def bayer_block_downsample(img: np.ndarray, k: int) -> np.ndarray:
    """Downsample a raw Bayer mosaic by block-averaging *within* each of its 4
    same-colour sub-planes independently, by a factor of ``k`` each, then
    re-interleaving into a smaller mosaic with the same 2x2 CFA pattern.

    Averaging across the raw mosaic directly (the naive block-average) would
    mix different sensor colour channels together -- R/G/B have very
    different absolute response (real gains, not just vignetting), so that
    would corrupt the result, not just blur it. Averaging within one colour
    at a time is what a flat's own low-rank content (vignetting, dust motes)
    survives fine: both vary smoothly over tens-to-hundreds of pixels, well
    above the few-pixel scale a modest ``k`` removes. Any remainder pixels
    (sub-plane dimension not divisible by ``k``) are cropped, not padded --
    at most ``k-1`` pixels per edge, negligible against a full sensor plane.
    """
    if k <= 1:
        return img
    planes_small = []
    for plane in _bayer_planes(img):
        h, w = plane.shape
        h2, w2 = h - h % k, w - w % k
        p = plane[:h2, :w2].reshape(h2 // k, k, w2 // k, k)
        planes_small.append(p.mean(axis=(1, 3)))
    sh, sw = planes_small[0].shape
    small = np.empty((sh * 2, sw * 2), dtype=planes_small[0].dtype)
    small[0::2, 0::2] = planes_small[0]
    small[0::2, 1::2] = planes_small[1]
    small[1::2, 0::2] = planes_small[2]
    small[1::2, 1::2] = planes_small[3]
    return small


def bayer_block_upsample(img_small: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Inverse of ``bayer_block_downsample``: upsample each of the 4 CFA
    sub-planes independently (bilinear -- a flat is smooth by assumption, so
    this is the same kind of interpolation the flat is later applied
    through) and re-interleave, cropped/edge-padded to ``target_shape``
    exactly (the downsample's remainder crop means the scale factor isn't
    always a perfectly round number)."""
    H, W = target_shape
    sub_h, sub_w = H // 2, W // 2
    out = np.empty((H, W), dtype=img_small.dtype)
    planes_small = _bayer_planes(img_small)
    for (r_off, c_off), p in zip([(0, 0), (0, 1), (1, 0), (1, 1)], planes_small):
        zoom_y = sub_h / p.shape[0]
        zoom_x = sub_w / p.shape[1]
        up = ndimage.zoom(p, (zoom_y, zoom_x), order=1)[:sub_h, :sub_w]
        if up.shape != (sub_h, sub_w):
            up = np.pad(up, ((0, sub_h - up.shape[0]), (0, sub_w - up.shape[1])), mode='edge')
        out[r_off::2, c_off::2] = up
    return out


def robust_pca_master(frames: List[FrameInfo], shape: Tuple[int, ...],
                      downsample: int = 1) -> Optional[np.ndarray]:
    """Build a master calibration frame via robust PCA.

    Returns the per-pixel median of the recovered low-rank component ``L``
    (which should already be near rank-1 once converged, so its median
    column is essentially the clean master), or ``None`` if too few frames
    loaded successfully, the stack won't fit in available memory, or any
    loaded frame is non-finite -- caller should fall back to a plain median
    in every ``None`` case.

    ``downsample`` (> 1): block-averages each loaded frame via
    ``bayer_block_downsample`` before decomposition and upsamples the
    recovered master back to ``shape`` via ``bayer_block_upsample``.
    Deliberately **not** used for real bias/dark/flat robust-PCA masters --
    only ``--flat-from-lights`` passes this, where the sparse component only
    needs to isolate stars/nebula structure that dithers between frames (not
    per-pixel hot pixels the way a real dark/bias decomposition does), and
    the low-rank component it's actually after (vignetting, dust motes) is
    smooth well above the pixel scale a modest downsample removes. Cuts P
    (and so the O(N^2 x P) decomposition cost) by ``downsample**2``.
    """
    from src.io_fits import load_frame

    # A CFA-respecting downsample only makes sense on a real 2x2-mosaic raw
    # frame; anything else (mono, already-debayered, odd dimensions) falls
    # back to full resolution rather than guessing at a layout.
    can_downsample = (downsample > 1 and len(shape) == 2
                      and shape[0] % 2 == 0 and shape[1] % 2 == 0)
    work_shape = shape

    imgs = []
    skipped = 0
    for f in frames:
        try:
            data, _ = load_frame(f.path)
        except Exception:
            continue
        if data.shape != shape:
            # Mixed binning/ROI within one calibration type -- np.stack below
            # requires uniform shape, and the caller's homogeneity fast-path
            # (select_matching_darks/flats) doesn't check dimensions, so a
            # mismatched frame can genuinely reach here.
            skipped += 1
            continue
        data = data.astype(np.float64)
        if can_downsample:
            data = bayer_block_downsample(data, downsample)
            work_shape = data.shape
        imgs.append(data)

    if skipped:
        _log.warning(
            "robust_pca_master: skipped %d frame(s) with shape mismatched to "
            "the reference frame (expected %s)", skipped, shape)

    if len(imgs) < Config.ROBUST_PCA_MIN_FRAMES:
        _log.warning(
            "robust_pca_master: only %d frames loaded (need >= %d for a "
            "meaningful low-rank/sparse split); falling back to median",
            len(imgs), Config.ROBUST_PCA_MIN_FRAMES)
        safe_print(f"  robust_pca: only {len(imgs)} usable frames "
                   f"(need >= {Config.ROBUST_PCA_MIN_FRAMES}) -- falling back to median")
        return None

    n = len(imgs)
    p = int(np.prod(work_shape))
    # IALM keeps ~7 live (n, p) float64 arrays alive at once (D, L, S, Y, temp,
    # residual, plus the SVD's working copies) -- guard against OOM the same
    # way make_master's median path guards its memmap threshold, since --auto
    # can route here with no awareness of sensor resolution.
    estimated_bytes = n * p * 8 * 7
    try:
        import psutil
        avail = psutil.virtual_memory().available
    except Exception:
        avail = None
    if avail is not None and estimated_bytes > avail // 2:
        _log.warning(
            "robust_pca_master: estimated memory use (%.1f GB) exceeds half of "
            "available RAM (%.1f GB); falling back to median",
            estimated_bytes / 1e9, avail / 1e9)
        safe_print(f"  robust_pca: stack too large for available memory "
                   f"(~{estimated_bytes/1e9:.1f} GB needed) -- falling back to median")
        return None

    stack = np.stack(imgs, axis=0)
    imgs = None  # drop the per-frame list before the decomposition's own peak usage
    D = stack.reshape(n, -1)

    if not np.all(np.isfinite(D)):
        _log.warning(
            "robust_pca_master: non-finite pixel values in calibration stack; "
            "falling back to median")
        safe_print("  robust_pca: non-finite pixel values detected -- falling back to median")
        return None

    L, _S = robust_pca_decompose(D)
    master = np.median(L, axis=0).reshape(work_shape)
    if work_shape != shape:
        master = bayer_block_upsample(master, shape)
    return master.astype(np.float32)
