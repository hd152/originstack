# OriginStack — Quick Reference

## One-Line Commands

```bash
# Basic stack with all defaults
python astro_stack.py -d lights/ -o stacked.fits

# With verbose per-frame output
python astro_stack.py -d lights/ -o stacked.fits -v

# Auto target detection (no API key needed)
python astro_stack.py -d lights/ -o stacked.fits -v --auto

# Specific target preset
python astro_stack.py -d lights/ -o stacked.fits --preset galaxy

# Keep all frames (disable quality filter)
python astro_stack.py -d lights/ -o stacked.fits --no-quality-filter

# Debug registration problems
python astro_stack.py -d lights/ -o stacked.fits --debug registration

# Health check without stacking
python astro_stack.py -d lights/ --health-check

# Dry run — see resolved parameters without processing
python astro_stack.py -d lights/ -o stacked.fits --dry-run

# Iterate fast on the SAME output — re-runs skip Phases 1-3 (redo post only)
python astro_stack.py -d lights/ -o stacked.fits --auto --keep-checkpoint

# Incremental stacking — fold a previous night's saved stack into this run
python astro_stack.py -d tonight/ -o m51_v2.fits --auto --merge m51.fits
```

### Optional native (Rust) acceleration
13 hot-path kernels run in Rust when the `astro_native` module is built
(stacking combines ~4–100×, Lanczos warp for alignment and drizzle ~5–26×,
L.A.Cosmic, median filters, the MMT median cascade ~10×, DBE sampling+fit,
anisotropic diffusion ~37×); otherwise a numpy fallback is used. Build once:

```bash
cd ext/astro_native && maturin develop --release           # into a venv
# or (system Python): python -m maturin build --release && pip install --force-reinstall target/wheels/astro_native-*.whl
```

The startup banner shows `Native accel: … ACTIVE`; accelerated steps log `[rust] …`.

---

## Understanding Verbose Output

### Quality Metrics

```
light_001.fit: brightness=18.2, contrast=12.1, stars=156, snr=42.1, score=0.87
               ↑                ↑               ↑          ↑         ↑
        Exposure level    Detail/dynamic    Star count   Signal   Composite
                          range indicator               to noise  quality
```

**Healthy ranges:**
| Metric | Good | Investigate |
|--------|------|-------------|
| Brightness | 15–22 | < 10 or > 28 |
| Contrast | 8–20 | < 5 |
| Stars | > 80 | < 30 |
| SNR | > 20 | < 10 |

### Shift Output

```
light_001.fit: shift=(+0.1, +2.1) px, magnitude=2.10 px
               ↑                       ↑
        X and Y pixel offsets     Total displacement
        (+ = right/down)          (Pythagorean distance)
```

**Shift magnitudes:**
| Magnitude | Meaning |
|-----------|---------|
| < 0.5 px | Excellent tracking |
| 0.5–5 px | Normal for unguided or lightly guided |
| 5–15 px | Check guiding; will be corrected by registration |
| > 15 px | Possible guide star loss or mount problem |

---

## Shift Pattern Guide

| Pattern | Likely Cause | Action |
|---------|-------------|--------|
| Small stable scatter ≤ 0.5 px | Perfect tracking | Nothing |
| Steady linear drift | Polar alignment or RA tracking rate | Tune mount |
| Oscillation (1–3 px repeating) | Wind, vibration, or periodic error | Check mount stability |
| Sudden large jump then return | Guide star loss, auto-guider correction | Review those frames |
| Increasing magnitude over time | Slow PE or Dec drift | Check alignment |
| High random scatter | Atmospheric seeing | Normal at high magnification |

---

## Diagnosing Problem Frames

**Scenario: One frame has low contrast AND a large shift**
```
light_001.fit: brightness=18.2, contrast=12.1, stars=156, shift=0.22 px  ← normal
light_002.fit: brightness=18.3, contrast=11.9, stars=163, shift=0.22 px  ← normal
light_003.fit: brightness=17.2, contrast= 8.3, stars= 87, shift=6.60 px  ← PROBLEM
light_004.fit: brightness=18.1, contrast=12.0, stars=161, shift=0.30 px  ← normal
```

Frame 003 likely had a focus adjustment or brief cloud. The quality filter rejects it automatically.

---

## Post-Processing Default Flags

| Feature | On by default | Disable |
|---------|:---:|---------|
| Background extraction (DBE) | ✅ | `--no-background-extraction` |
| Luma denoising (wavelet) | ✅ | `--denoiser none` |
| Chroma noise reduction | ✅ | `--no-chroma-nr` |
| Star reduction | ✅ | `--no-star-reduce` |
| Local contrast enhancement | ✅ | `--no-local-contrast` |
| CA correction | ✅ | `--no-ca-correction` |
| Cosmic ray rejection | auto | `--no-cosmic-ray-rejection` (auto-skipped on deep rejection stacks) |
| Quality filtering | ✅ | `--no-quality-filter` |
| Affine registration | ✅ | `--no-affine` |

| Feature | Off by default | Enable |
|---------|:---:|---------|
| Alternative primary denoiser | ❌ | `--denoiser {mmt,bm3d,acdnr,nlm,bilateral,aniso}` |
| Deconvolution | ❌ | `--deconvolve {rl,tv}` (RL on GPU with `--use-gpu`) |
| Coarse chroma-NR (colour blotches) | auto | config key `chroma_nr_large_sigma` |
| Preview black point (sky-σ) | auto | `--preview-black-sigma 3` |
| Drizzle super-resolution | ❌ | `--drizzle-scale 2.0` |
| Plate solving | ❌ | `--plate-solve` |
| Star removal | ❌ | `--star-remove` |

---

## Presets

```bash
--preset quick        # Fastest: mean stack, minimal post-processing
--preset quality      # Best output: sigma_clip, all denoisers, deconvolution
--preset galaxy       # GHS stretch, star reduction, bilateral
--preset nebula       # GHS stretch, MMT + ACDNR
--preset narrowband   # Tuned for Ha/OIII/SII data
--preset starfield    # No star reduction, minimal processing
--preset planetary    # No background, with deconvolution
--preset lunar        # Linear stretch, no star reduction
```

---

## Collecting a Diagnostic Log

```bash
python astro_stack.py -d lights/ -o out.fits -v 2>&1 | tee diagnostic.log
```

The `diagnostic.log` file contains:
- Per-frame quality metrics and shift values
- Reference frame selection
- Frame rejection reasons
- Stacking configuration
- Phase timing

---

## Common Questions

**Q: How do I know which frames were rejected?**  
Run with `-v`. Each rejected frame will show `[REJECTED]` with the reason.

**Q: Can I re-run post-processing without re-stacking?**  
Yes. Use `--keep-checkpoint` on the first run, then on subsequent runs the raw stack is loaded from the checkpoint automatically.

**Q: My shifts are all > 10 px — is that a problem?**  
Not necessarily. Registration will correct them. Large shifts only matter if they exceed the image border (> ~50 px for typical setups), which would cause the valid crop to cut off too much. Use `--debug registration` to inspect.

**Q: Why does the preview JPEG look different from the FITS?**  
The FITS is stored as linear float32 data. The JPEG has a stretch applied (GHS by default). Load the FITS in PixInsight, Siril, or AstroImageJ and apply your own stretch.

**Q: Frames look OK but post-processing is too aggressive?**  
Try `--no-star-reduce --no-local-contrast` first to isolate which step is causing issues. Then use `--debug diagnostic` to get a FITS snapshot before each step.

---

## Further Documentation

- [README.md](README.md) — Installation, quick start, examples
- [PROJECT_SPEC.md](PROJECT_SPEC.md) — Full CLI reference and architecture detail
