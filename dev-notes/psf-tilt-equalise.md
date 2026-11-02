# PSF tilt equalisation (`--deconvolve nn` + `--config nn_tilt_equalise=true`)

Every Origin stack measured has a clean linear gradient of star FWHM along one sensor axis
([stack-optics](stack-optics.md)). `--deconvolve nn` flattens most of it but not all.
`src/psf_field.equalise_tilt` runs the network once more per strip along the gradient and
blends each strip's result in only as far as it takes the strip's star width to the sharpest
strip's width (measured on the strip's own stars before and after).

Tried first (not kept): (a) a classical RL pass with `relative_kernel` per strip -- flattened it but
rang around faint stars; (b) the network pass at a strength predicted from an assumed width ratio
(0.78 per pass) -- overshot on Pleiades (left 1.70 px against 2.11 px on the right, below the
network's trained 2.3 px minimum). The measured blend replaced it.

Five held-out stacks, Phase 4 with `--deconvolve nn`, star FWHM by column (left to right), without
-> with the pass:

| Stack | NN only | NN + equalise | Aperture flux ratio (r = 9 px, R/G/B) | Sky noise | 24 px-smoothed luminance, p1-p99 |
|---|---|---|---|---|---|
| Omega | 3.44 3.45 3.13 2.58 (1.33) | 2.29 2.39 2.38 2.44 (1.07) | 0.989 / 0.996 / 0.995 | 42.9 -> 40.2 | 0.9953-0.9985 |
| Orion | 5.41 4.58 3.92 2.70 (2.00) | 3.87 3.52 3.08 2.53 (1.53) | 0.993 / 0.990 / 0.981 | 13.4 -> 12.1 | 0.9886-0.9985 |
| Pleiades | 3.40 2.73 2.44 2.14 (1.59) | 2.00 2.13 2.16 2.14 (1.08) | 0.997 / 1.000 / 0.998 | 6.8 -> 6.4 | 0.9980-0.9992 |
| Sunflower | 1.86 1.78 1.68 1.66 (1.12) | 1.72 1.70 1.66 1.62 (1.06) | 0.991 / 0.994 / 0.993 | 8.7 -> 8.3 | 0.9968-0.9980 |
| Sculptor (tilt along y) | -- | small change | 0.994 / 0.994 / 1.000 | 6.7 -> 6.4 | 0.9989-1.0001 |

(max/min of the column medians in brackets.) Star colour index did not shift (Omega, first version:
-0.001). On Orion the network cannot narrow the wide strips enough in one more pass (blends
saturate at 1.0), so 1.53 remains.

Cost: one extra network run over the strips that need it, about 15-35 s on these stacks. Off by
default (`nn_tilt_equalise=false`).

Limits: five stacks, one camera unit; a second pass on the network's own output is outside its
training distribution (the checks above are flux, colour, large-scale light and visual
inspection of crops, not a truth comparison); the strip widths come from unsaturated stars brighter
than 25 sigma, so a star-poor strip falls back to no change.
