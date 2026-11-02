# Gaia-anchored halo subtraction (`--halo-removal` with `halo_removal_method=psf`)

`src/halo_model.py` subtracts bright-star wings from the linear stack. The old `--halo-removal`
(`denoising.remove_star_halos`, one Gaussian per star) is a no-op on real stacks: Pleiades preview
unchanged to 0.02/255 in rings of 40-220 px around the brightest star.

Method (after Infante-Sainz, Trujillo & Roman 2020, arXiv:1911.01430), per channel:
1. flux scale `log10 F = a - 0.4 G + b (BP-RP)` from unsaturated Gaia stars (aperture photometry);
   scatter 0.02-0.04 dex on Pleiades, Orion, Omega, Sunflower once the aperture is sized from a
   sane FWHM;
2. radial PSF per unit flux: inverse-variance weighted least squares over the reference stars,
   radius by radius (a median over the bins that came out above 4 sigma is biased high -- first
   version, over-subtracted Orion and Omega by up to 46 sigma);
3. subtract `0.85 * F * PSF(r)` beyond `r_in = 2.5 FWHM` (FWHM capped at 6 px; the measured stack
   FWHM is 9 px on Pleiades because the bright halos dominate the sample), capped per angular sector at
   the light actually present there (12 sectors, the weaker of each sector and its neighbours).

The cap matters: the Alcyone halo is not radially symmetric and the uncapped model dug a 9-sigma
moat in the weak direction (visible as a black disc in the preview). With it, 0.2% of the pixels
within 120 px are more than 5 sigma below sky (none without the step).

Measured (halo excess above local sky in pixel-noise sigmas, brightest isolated stars, before ->
after, at r = 20 / 40 / 80 px): Pleiades 96-108 -> 40-65 / 16-23 -> 4-8 / 1.8-3.3 -> 0.7-2.0;
Orion (nebula-embedded star) 189/93/49 -> 185/92/49 (the nebula is not mistaken for halo: the
amplitude comes from the catalogue); Omega, Sunflower: halos already under 1-2 sigma, unchanged.
Star cores at r = 10 px are untouched (74.6 -> 74.3). Injected faint cirrus (`tools/bench_lsb_injection.py`
field) passes the step with T = 0.98-1.03 at every scale 16-1024 px.

Limits: Gaia is incomplete for G < ~7 and the model skips stars without a catalogue entry; one
radial PSF per channel (no spikes, no reflection rings; the inner 20-50 px of the brightest star
keeps radial streaks and a ring); the far wing (> 70 px) is under-subtracted (the window edge and
the local sky baseline absorb some). Needs the stack WCS and a Gaia cone (cached locally, so a
`--from-stack` rerun works offline); declines with a note and falls back to the Gaussian remover.
