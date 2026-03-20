"""AI-powered parameter advisor and session report generator.

Requires the ``anthropic`` package and ``ANTHROPIC_API_KEY`` environment
variable.  Both features degrade gracefully when either is absent.

Usage
-----
Parameter advisor (runs after Phase 1, before stacking):
    from src.ai_advisor import get_parameter_recommendations, apply_recommendations
    rec, explanation = get_parameter_recommendations(final, rejected_reasons, args)
    if rec:
        changes = apply_recommendations(rec, args)

Session report (runs after Phase 4):
    from src.ai_advisor import build_report_context, generate_session_report
    ctx = build_report_context(final, rejected_reasons, args, stats, shifts, dither_info,
                               output_path, stacked_shape)
    generate_session_report(ctx, output_path)
"""
from __future__ import annotations

import json
import os
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import argparse

_MODEL = "claude-opus-4-6"

try:
    import anthropic
    from pydantic import BaseModel, Field
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    BaseModel = object  # stub so the class body is valid at parse time


# ---------------------------------------------------------------------------
# Structured output schema for the parameter advisor
# ---------------------------------------------------------------------------

if HAS_ANTHROPIC:
    class _Recommendations(BaseModel):
        """Per-parameter recommendations from the AI advisor."""
        stack_method: Optional[str] = Field(
            None,
            description=(
                "Stacking method to use: sigma_clip, winsorized, percentile, esd, "
                "median, mean, or null to keep the current value."
            ),
        )
        rejection_sigma: Optional[float] = Field(
            None,
            description="Sigma rejection threshold (2.0–4.0), or null to keep current.",
        )
        rejection_iters: Optional[int] = Field(
            None,
            description="Sigma-clip iterations (2–5), or null to keep current.",
        )
        drizzle_scale: Optional[float] = Field(
            None,
            description=(
                "Drizzle super-resolution scale factor (1.5 or 2.0), or null/1.0 to "
                "disable drizzle. Only recommend > 1.0 when dithering is confirmed."
            ),
        )
        denoise_strength: Optional[float] = Field(
            None,
            description="Wavelet denoising threshold factor (1.5–6.0), or null to keep current.",
        )
        explanation: str = Field(
            ...,
            description=(
                "Plain-language explanation of every recommendation (or confirmation "
                "that current settings are appropriate). Be specific and cite the numbers."
            ),
        )
        warnings: list[str] = Field(
            default_factory=list,
            description="Any concerns or caveats about the data quality or session.",
        )


# ---------------------------------------------------------------------------
# Advisor: recommend stacking parameters from Phase 1 quality stats
# ---------------------------------------------------------------------------

_ADVISOR_SYSTEM = """\
You are an expert astrophotography image-stacking advisor embedded in an automated pipeline.

After Phase 1 (quality analysis) you receive per-frame statistics and must recommend optimal
stacking parameters for Phases 2–4.  Only recommend changing a parameter when the data clearly
supports it.  When current settings are appropriate, say so and leave the corresponding field null.

Key heuristics (apply judiciously — not mechanically):
• Frame count < 8:  prefer 'percentile' or 'esd' (sigma-clip MAD is unreliable for small N).
• Frame count 8–20: sigma_clip, sigma 3.0–3.5.
• Frame count > 20: sigma_clip, sigma 2.5–3.0.  More iterations (3–5) are safe.
• High FWHM spread (std > 1.5 px): tighter sigma (2.5) or winsorize to down-weight bad frames.
• Low SNR (mean < 5): increase denoise_strength toward 4–5 to suppress noise.
• High SNR (mean > 20): lower denoise_strength to 1.5–2 to preserve detail.
• Dithered data: drizzle_scale 1.5–2.0 recovers resolution; only recommend if n_frames >= 10.
• Rejection breakdown "No stars detected" > 20% of rejects: seeing was unstable; use sigma 3.5.
• Acceptance rate > 95%: quality gate is healthy, no changes needed.
• Acceptance rate < 50%: data quality is marginal; add a warning.
"""


def _build_advisor_context(final: list, rejected_reasons: dict, args) -> dict:
    """Distil Phase 1 results into the feature dict passed to the LLM."""
    scores = np.array([f.metrics.get("score", 0.0) for f in final], dtype=float)
    fwhms = [f.metrics.get("fwhm", 0.0) for f in final if f.metrics.get("fwhm", 0.0) > 0]
    snrs = [f.metrics.get("snr", 0.0) for f in final]
    star_counts = [f.metrics.get("star_count", 0) for f in final]

    def _stats(vals: list) -> dict:
        if not vals:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        a = np.array(vals, dtype=float)
        return {
            "mean": round(float(np.mean(a)), 2),
            "std": round(float(np.std(a)), 2),
            "min": round(float(np.min(a)), 2),
            "max": round(float(np.max(a)), 2),
        }

    # Categorise raw rejection strings into human-friendly buckets
    reason_counts: dict[str, int] = {}
    for reason in rejected_reasons.values():
        if "score" in reason:
            cat = "Below quality threshold"
        elif "outlier" in reason:
            cat = "Statistical outlier"
        elif any(k in reason for k in ("brightness", "contrast", "dynamic", "noise")):
            cat = "Poor image quality"
        elif "star" in reason:
            cat = "No stars detected"
        elif any(k in reason for k in ("load", "empty", "error")):
            cat = "Load / data error"
        else:
            cat = "Other"
        reason_counts[cat] = reason_counts.get(cat, 0) + 1

    n_total = len(final) + len(rejected_reasons)
    return {
        "n_frames_total": n_total,
        "n_frames_accepted": len(final),
        "n_frames_rejected": len(rejected_reasons),
        "acceptance_rate_pct": round(100.0 * len(final) / max(n_total, 1), 1),
        "rejection_breakdown": reason_counts,
        "quality_scores": _stats(scores.tolist()),
        "fwhm_pixels": _stats(fwhms),
        "snr": _stats(snrs),
        "star_counts": _stats(star_counts),
        "current_settings": {
            "stack_method": args.stack_method,
            "rejection_sigma": args.rejection_sigma,
            "rejection_iters": args.rejection_iters,
            "drizzle_scale": getattr(args, "drizzle_scale", 1.0),
            "denoise": getattr(args, "denoise", True),
            "denoise_strength": getattr(args, "denoise_strength", 3.0),
        },
    }


def _format_advisor_context(ctx: dict) -> str:
    lines = [
        f"Frames: {ctx['n_frames_accepted']} accepted / {ctx['n_frames_total']} total "
        f"({ctx['acceptance_rate_pct']}% acceptance rate, {ctx['n_frames_rejected']} rejected)",
    ]
    if ctx["rejection_breakdown"]:
        lines.append(
            "Rejection breakdown: "
            + ", ".join(f"{k}: {v}" for k, v in ctx["rejection_breakdown"].items())
        )
    qs = ctx["quality_scores"]
    lines.append(
        f"Quality scores: mean={qs['mean']:.1f}, std={qs['std']:.1f}, "
        f"min={qs['min']:.1f}, max={qs['max']:.1f}"
    )
    fw = ctx["fwhm_pixels"]
    if fw["mean"] > 0:
        lines.append(
            f"FWHM: mean={fw['mean']:.2f}px, std={fw['std']:.2f}px, "
            f"min={fw['min']:.2f}px, max={fw['max']:.2f}px"
        )
    snr = ctx["snr"]
    lines.append(f"SNR: mean={snr['mean']:.1f}, std={snr['std']:.1f}")
    sc = ctx["star_counts"]
    lines.append(f"Stars per frame: mean={sc['mean']:.0f}, std={sc['std']:.0f}")
    cs = ctx["current_settings"]
    lines.append(
        f"Current settings: stack_method={cs['stack_method']}, "
        f"rejection_sigma={cs['rejection_sigma']}, rejection_iters={cs['rejection_iters']}, "
        f"drizzle_scale={cs['drizzle_scale']}, "
        f"denoise={cs['denoise']}, denoise_strength={cs['denoise_strength']}"
    )
    return "\n".join(lines)


def get_parameter_recommendations(final: list, rejected_reasons: dict, args):
    """Call Claude to recommend stacking parameters from Phase 1 quality stats.

    Returns ``(recommendations, explanation_text)`` on success, or
    ``(None, None)`` when the feature is unavailable or the call fails.
    """
    if not HAS_ANTHROPIC:
        print(
            "  [AI advisor] anthropic not installed — skipping. "
            "Run: pip install anthropic"
        )
        return None, None

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  [AI advisor] ANTHROPIC_API_KEY not set — skipping.")
        return None, None

    ctx = _build_advisor_context(final, rejected_reasons, args)
    prompt = (
        "Here are the Phase 1 quality statistics for this astrophotography session:\n\n"
        + _format_advisor_context(ctx)
        + "\n\nRecommend optimal stacking parameters. "
        "Only suggest changes where the data clearly supports them."
    )

    print("\n  Consulting AI parameter advisor...", end="", flush=True)
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.parse(
            model=_MODEL,
            max_tokens=2048,
            system=_ADVISOR_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_format=_Recommendations,
        )
        rec = response.parsed_output
        print(" done.")
        return rec, rec.explanation
    except Exception as exc:
        print(f" failed ({type(exc).__name__}: {exc})")
        return None, None


def apply_recommendations(rec, args) -> list[str]:
    """Apply AI recommendations to ``args`` in-place.

    Returns a list of human-readable change descriptions.
    """
    changes: list[str] = []

    if rec.stack_method and rec.stack_method != args.stack_method:
        args.stack_method = rec.stack_method
        changes.append(f"stack_method  {args.stack_method!r} → {rec.stack_method!r}")

    if rec.rejection_sigma is not None and rec.rejection_sigma != args.rejection_sigma:
        old = args.rejection_sigma
        args.rejection_sigma = rec.rejection_sigma
        changes.append(f"rejection_sigma  {old} → {rec.rejection_sigma}")

    if rec.rejection_iters is not None and rec.rejection_iters != args.rejection_iters:
        old = args.rejection_iters
        args.rejection_iters = rec.rejection_iters
        changes.append(f"rejection_iters  {old} → {rec.rejection_iters}")

    if rec.drizzle_scale is not None:
        current = getattr(args, "drizzle_scale", 1.0)
        if abs(rec.drizzle_scale - current) > 0.01:
            args.drizzle_scale = rec.drizzle_scale
            changes.append(f"drizzle_scale  {current} → {rec.drizzle_scale}")

    if rec.denoise_strength is not None:
        current = getattr(args, "denoise_strength", 3.0)
        if abs(rec.denoise_strength - current) > 0.01:
            args.denoise_strength = rec.denoise_strength
            changes.append(f"denoise_strength  {current} → {rec.denoise_strength}")

    return changes


# ---------------------------------------------------------------------------
# Session report: narrative summary of the completed stack
# ---------------------------------------------------------------------------

_REPORT_SYSTEM = """\
You are an expert astrophotography image-processing advisor writing a session report.

You have complete statistics from a finished stacking run.  Write a clear, informative
Markdown report with these five sections:

1. **Session Summary** — one-paragraph overview of what was processed and overall quality.
2. **Frame Quality** — acceptance rate, key metrics, notable rejection patterns.
3. **Registration** — how well the frames aligned; any concerns (large shifts, identical shifts, etc.).
4. **Stacking & Post-processing** — method used, whether it suits the data, denoising outcome.
5. **Recommendations for Next Session** — three to five concrete, actionable suggestions
   (e.g. more frames, longer exposure, dithering, better calibration, cooler sensor, etc.).

Rules:
• Be specific — cite the actual numbers from the data.
• Avoid generic filler.  Every sentence should convey useful information.
• Keep the total length to roughly 400–600 words.
• Use Markdown formatting (bold headings, bullet lists where appropriate).
"""


def build_report_context(
    final: list,
    rejected_reasons: dict,
    args,
    stats,
    shifts: list,
    dither_info: dict,
    output_path: str,
    stacked_shape: tuple,
) -> dict:
    """Build the context dict passed to the session report LLM."""
    scores = [f.metrics.get("score", 0.0) for f in final]
    fwhms = [f.metrics.get("fwhm", 0.0) for f in final if f.metrics.get("fwhm", 0.0) > 0]
    snrs = [f.metrics.get("snr", 0.0) for f in final]
    star_counts = [f.metrics.get("star_count", 0) for f in final]

    def _stats(vals: list) -> dict:
        if not vals:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        a = np.array(vals, dtype=float)
        return {
            "mean": round(float(np.mean(a)), 2),
            "std": round(float(np.std(a)), 2),
            "min": round(float(np.min(a)), 2),
            "max": round(float(np.max(a)), 2),
        }

    # Rejection breakdown
    reason_counts: dict[str, int] = {}
    for reason in rejected_reasons.values():
        if "score" in reason:
            cat = "Below quality threshold"
        elif "outlier" in reason:
            cat = "Statistical outlier"
        elif any(k in reason for k in ("brightness", "contrast", "dynamic", "noise")):
            cat = "Poor image quality"
        elif "star" in reason:
            cat = "No stars detected"
        elif any(k in reason for k in ("load", "empty", "error")):
            cat = "Load / data error"
        else:
            cat = "Other"
        reason_counts[cat] = reason_counts.get(cat, 0) + 1

    # Registration statistics from shifts
    shift_mags = [float(np.sqrt(sy**2 + sx**2)) for sy, sx in shifts]
    reg_stats = {
        "mean_magnitude_px": round(float(np.mean(shift_mags)), 2) if shift_mags else 0.0,
        "max_magnitude_px": round(float(np.max(shift_mags)), 2) if shift_mags else 0.0,
        "std_magnitude_px": round(float(np.std(shift_mags)), 2) if shift_mags else 0.0,
        "frames_with_zero_shift": sum(1 for m in shift_mags if m < 0.5),
    }

    n_total = len(final) + len(rejected_reasons)
    oh, ow = stacked_shape[:2]

    return {
        "output_path": output_path,
        "output_size": f"{oh}×{ow}",
        "n_total": n_total,
        "n_accepted": len(final),
        "n_rejected": len(rejected_reasons),
        "acceptance_rate_pct": round(100.0 * len(final) / max(n_total, 1), 1),
        "rejection_breakdown": reason_counts,
        "quality_scores": _stats(scores),
        "fwhm_pixels": _stats(fwhms),
        "snr": _stats(snrs),
        "star_counts": _stats(star_counts),
        "registration": reg_stats,
        "dither_info": {
            "pattern": dither_info.get("pattern", "unknown"),
            "mean_magnitude_px": round(dither_info.get("mean_magnitude", 0.0), 2),
            "unique_positions": dither_info.get("unique_positions", 0),
        },
        "settings": {
            "stack_method": args.stack_method,
            "rejection_sigma": args.rejection_sigma,
            "rejection_iters": args.rejection_iters,
            "drizzle_scale": getattr(args, "drizzle_scale", 1.0),
            "denoise": getattr(args, "denoise", True),
            "denoise_strength": getattr(args, "denoise_strength", 3.0),
            "background_extraction": getattr(args, "background_extraction", True),
            "debayer_method": args.debayer_method,
        },
        "timing": {
            "quality_analysis_s": round(stats.quality_time, 1),
            "registration_s": round(stats.registration_time, 1),
            "stacking_s": round(stats.stacking_time, 1),
            "post_processing_s": round(stats.post_processing_time, 1),
            "total_s": round(stats.total_time(), 1),
        },
    }


def _format_report_context(ctx: dict) -> str:
    lines = [
        f"**Output file:** {os.path.basename(ctx['output_path'])}  ({ctx['output_size']} pixels)",
        f"**Frames:** {ctx['n_accepted']} stacked / {ctx['n_total']} analyzed "
        f"({ctx['acceptance_rate_pct']}% acceptance, {ctx['n_rejected']} rejected)",
    ]
    if ctx["rejection_breakdown"]:
        lines.append(
            "**Rejection reasons:** "
            + ", ".join(f"{k}: {v}" for k, v in ctx["rejection_breakdown"].items())
        )

    qs = ctx["quality_scores"]
    lines.append(
        f"**Quality scores:** mean={qs['mean']:.1f}, std={qs['std']:.1f}, "
        f"range [{qs['min']:.1f}–{qs['max']:.1f}]"
    )
    fw = ctx["fwhm_pixels"]
    if fw["mean"] > 0:
        lines.append(
            f"**FWHM:** mean={fw['mean']:.2f}px, std={fw['std']:.2f}px "
            f"(range {fw['min']:.2f}–{fw['max']:.2f}px)"
        )
    snr = ctx["snr"]
    lines.append(f"**SNR:** mean={snr['mean']:.1f}, std={snr['std']:.1f}")
    sc = ctx["star_counts"]
    lines.append(f"**Stars per frame:** mean={sc['mean']:.0f}, std={sc['std']:.0f}")

    reg = ctx["registration"]
    lines.append(
        f"**Registration:** mean shift {reg['mean_magnitude_px']:.1f}px, "
        f"max {reg['max_magnitude_px']:.1f}px, std {reg['std_magnitude_px']:.1f}px; "
        f"{reg['frames_with_zero_shift']} frames with near-zero shift"
    )

    di = ctx["dither_info"]
    lines.append(
        f"**Dither pattern:** {di['pattern']}, mean offset {di['mean_magnitude_px']:.1f}px, "
        f"{di['unique_positions']} unique positions"
    )

    s = ctx["settings"]
    lines.append(
        f"**Settings:** stack_method={s['stack_method']}, "
        f"rejection_sigma={s['rejection_sigma']}, rejection_iters={s['rejection_iters']}, "
        f"drizzle_scale={s['drizzle_scale']}, denoise_strength={s['denoise_strength']}, "
        f"debayer={s['debayer_method']}, background_extraction={s['background_extraction']}"
    )

    t = ctx["timing"]
    lines.append(
        f"**Processing time:** {t['total_s']}s total "
        f"(quality {t['quality_analysis_s']}s, registration {t['registration_s']}s, "
        f"stacking {t['stacking_s']}s, post-processing {t['post_processing_s']}s)"
    )

    return "\n".join(lines)


def generate_session_report(ctx: dict, output_path: str) -> Optional[str]:
    """Stream a session report from Claude and save it alongside the output FITS.

    Prints the report live to stdout as it streams.  Returns the full report
    text on success, or ``None`` when the feature is unavailable.
    """
    if not HAS_ANTHROPIC:
        return None

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    prompt = (
        "Please write a session report for this astrophotography stacking run:\n\n"
        + _format_report_context(ctx)
    )

    print("\n" + "=" * 70)
    print("AI SESSION REPORT")
    print("=" * 70 + "\n")

    try:
        client = anthropic.Anthropic(api_key=api_key)
        chunks: list[str] = []

        with client.messages.stream(
            model=_MODEL,
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=_REPORT_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
                chunks.append(text)

        report = "".join(chunks)
        print("\n")

        # Save alongside the output FITS
        report_path = os.path.splitext(output_path)[0] + "_report.md"
        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write("# Astrophotography Session Report\n\n")
            fh.write(report)
        print(f"  Report saved: {os.path.basename(report_path)}")

        return report

    except Exception as exc:
        print(f"\n  [AI report] Error generating report: {type(exc).__name__}: {exc}")
        return None
