# What the optics leave in a stack: PSF tilt and lateral CA

`tools/stack_optics_census.py STACK.fits ...` measures, on the linear stack, the star FWHM in a 4x4
grid (adaptive Gaussian-weighted moments; `quality.measure_fwhm` is quantised to 1.732 steps and
made three stacks read a max/min of exactly 1.732), a plane fit of it, the R/G/B widths and the
R-G and B-G centroid offsets against position (lateral chromatic aberration).

2026-11-02, five held-out Origin stacks (cell FWHM in px, columns left to right; rows are flat):

| Stack | FWHM by column | Tilt across field | Plane-fit residual | R-G corner shift |
|---|---|---|---|---|
| Sunflower | 4.02 3.70 3.39 3.17 | 1.14 px | 1.2% | 0.15 px |
| Omega | 5.11 4.64 3.98 3.30 | 2.40 px | 1.6% | 0.25 px |
| Orion | 5.64 5.27 4.88 4.06 | 2.23 px | 2.5% | 0.21 px |
| Pleiades | 5.22 4.70 4.09 3.41 | 2.44 px | 1.4% | 0.21 px |
| Sculptor | 3.26 3.05 2.92 2.90 | 0.51 px | 1.9% | 0.16 px |

**A clean linear FWHM gradient along the sensor x axis, same direction in all five stacks** (plane
direction -179 to -163 degrees; across nights and targets): the left edge is 12-55% wider than
the right. The amplitude changes between sessions (focus), the direction does not: sensor tilt or
field curvature of this unit. Rows are flat. B/G width ratio 0.87-1.09 and R/G 0.93-0.99 (noisy;
no consistent colour trend). R-G lateral shift at the corner 0.15-0.25 px (B-G under 0.1): at the
report's 0.15 px gate, small against a 3-5 px FWHM.

**The NN deconvolver already flattens most of it**, but not all (Omega, `--deconvolve nn`): columns
5.11 4.64 3.98 3.30 -> 3.42 3.46 3.13 2.57 (max/min 1.55 -> 1.33). Without it (the default wavelet
path) the gradient is untouched (5.07 4.70 4.04 3.37).

Consequences:
- A position-dependent PSF *is* warranted by the stack (the gate was 1.05; it is 1.16-1.57). The
  per-frame, per-tile proper coadd was not (frame-by-cell scatter, [psf-census](psf-census.md));
  the variation is fixed to the sensor, not frame to frame.
- Worth trying next: a column-adaptive equalising pass after the NN deconvolver (a second pass
  with the relative kernel, `psf_deconvolution.relative_kernel`, from the local PSF to the median
  width), or `--deconvolve-svpsf` with the NN. Not built: noise amplification on the soft side
  (deconvolving 5.4 -> 3.5 px needs gains of 20+ at 0.35 cycles/px) and untested artifacts.
- A radial CA model is marginal at 0.15-0.25 px; not built.

Caveats: five stacks, one camera unit; cells with fewer than six unsaturated stars are dropped;
widths are of unsaturated stars brighter than 25 sigma.
