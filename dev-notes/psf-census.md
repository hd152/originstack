# PSF census: is tile-aware proper coaddition worth building?

`tools/psf_census.py SESSION_DIR` measures the FWHM of the stars in each cell of a 2x2 grid in
40 calibrated frames (sensor coordinates), removes each frame's mean (its seeing) and each cell's
mean (the field's fixed shape), and compares what is left with the measurement noise (stars split
in two halves). The residual is the "lucky patch" signal tile-aware proper coadd (a PSF and weight
per frame *and tile*) would use.

2026-11-02, Origin178, six local sessions, 2x2 cells (the Antares session had 3 usable frames):

| Session | Frames | Mean FWHM (px) | Frame-by-cell excess | Static max/min cell FWHM |
|---|---|---|---|---|
| Sunflower 2026-03-12 | 40 | 3.62 | 2.2% | 1.17 |
| Crab 2026-03-12 | 40 | 5.34 | 1.3% | 1.19 |
| Flaming Star 2026-04-10 | 39 | 4.33 | 3.7% | 1.38 |
| Markarian's Chain 2026-04-27 | 40 | - | 3.2% | 1.32 |
| Whirlpool 2026-04-06 | 13 | - | 0.0% | 1.31 |
| Black Eye 2026-03-14 | 37 | - | 1.3% | 1.29 |

Gate set by the research report: build only if the excess reaches 3%. Two of six sessions clear it,
by 0.2 and 0.7 points, on noise estimates of 2-4%. A frame-by-tile excess of 3% in FWHM is 6% in
FWHM^2, so per-tile weights would differ by about +-6% between frames; the sensitivity gain of
weighting by quality that varies by that much is about 1 + (6%)^2, roughly 0.4% (an estimate, not
a measurement). **Not built.**

The static cell-to-cell spread (1.17-1.38) is large in every session. It is in sensor coordinates:
the stack's field rotation (alt-az) averages the sensor's fixed shape, so the stack PSF is rounder
than any single frame's. Whether the *stack* has a radial FWHM trend worth a post-coadd correction
(report item "PSF homogenisation of star footprints") is a separate question this census does not
answer; run `--aberration-report` on a stack for that.

Caveats: 2x2 cells are coarse (3x3 needs more stars per cell than short sessions have); the
cell FWHM uses `quality.measure_fwhm` on the brightest stars of each cell, so cell-to-cell
differences also contain differences in star brightness and local background.
