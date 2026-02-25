# Diagnostics and Shift Robustness Improvements

## Overview
Enhanced the astrophotography FITS stacker with comprehensive diagnostic output and more robust shift calculation to help users understand frame quality and debug registration issues.

## Key Improvements

### 1. Enhanced Quality Analysis Output
When running with `-v/--verbose` flag, the stacker now displays per-frame quality metrics during the quality analysis phase:

```
Quality analysis: analyzing 6 light frames...
  light_000.fit: brightness=17.5, contrast=11.4, stars=177
  light_001.fit: brightness=17.5, contrast=11.5, stars=185
  ...
```

**Metrics displayed:**
- **brightness**: Median pixel value (indicates exposure)
- **contrast**: Standard deviation (indicates detail and dynamic range)
- **stars**: Number of detected star-like features
- **score**: brightness × contrast (combined quality metric)
- **SNR** (Signal-to-Noise Ratio): Available in the underlying metrics for further analysis

### 2. Shift Reporting with Diagnostics
During image registration, the stacker now reports calculated shifts for each frame:

```
Registration: calculating shifts for 6 frames (reference: light_001.fit)
  light_000.fit: shift=(+0.1, +2.1) px, magnitude=2.10 px
  light_001.fit: shift=(+0.0, +0.0) px, magnitude=0.00 px
  light_002.fit: shift=(-4.1, +4.1) px, magnitude=5.80 px
  ...
```

**Shift information:**
- **(x, y) shift**: Pixel offsets in X and Y directions (signed values show direction)
- **magnitude**: Total shift distance in pixels (helps identify outliers)

### 3. Improved Shift Calculation Robustness

#### Phase Correlation Stability
- **Overflow warning suppression**: NumPy overflow warnings are now suppressed in `phase_cross_correlation()`
- **Shift magnitude validation**: Calculated shifts are validated to be physically realistic (<50% of image size)
- **Finite checks**: Ensures shift values are not NaN or infinity before acceptance

#### Enhanced Centroid Fallback
The fallback registration method (used when phase correlation fails) is more stable:
- Uses **90th percentile threshold** (less aggressive than previous 95th)
- Requires **minimum 10 bright pixels** (prevents failure on low-signal regions)
- Better handles images with varying brightness and noise levels

## Usage

### View Quality Analysis Metrics
```bash
python astro_stack.py -d your_images/ -o output.fits -v
```

### Example Output
```
Processing target: my_observation
  Found 10 FITS files: 8 lights, 2 darks, 0 flats, 0 bias
  Quality analysis: analyzing 8 light frames...
    Object_001.fit: brightness=18.2, contrast=12.1, stars=156
    Object_002.fit: brightness=18.3, contrast=11.9, stars=163
    Object_003.fit: brightness=18.1, contrast=12.2, stars=159
    Object_004.fit: brightness=18.4, contrast=11.8, stars=161
    Object_005.fit: brightness=17.8, contrast=12.0, stars=158
    Object_006.fit: brightness=18.2, contrast=12.1, stars=160
    Object_007.fit: brightness=18.3, contrast=11.9, stars=162
    Object_008.fit: brightness=18.0, contrast=12.0, stars=159
  Registration: calculating shifts for 8 frames (reference: Object_004.fit)
    Object_001.fit: shift=(+1.2, +3.4) px, magnitude=3.61 px
    Object_002.fit: shift=(-0.8, +2.1) px, magnitude=2.28 px
    Object_003.fit: shift=(+0.5, +1.2) px, magnitude=1.30 px
    Object_004.fit: shift=(+0.0, +0.0) px, magnitude=0.00 px
    Object_005.fit: shift=(-1.5, -2.3) px, magnitude=2.73 px
    Object_006.fit: shift=(+1.0, +2.8) px, magnitude=2.99 px
    Object_007.fit: shift=(-0.3, -1.1) px, magnitude=1.14 px
    Object_008.fit: shift=(+0.8, +1.9) px, magnitude=2.07 px
Saved stacked FITS to output.fits and preview to output.jpg
```

## Interpreting Diagnostics

### Quality Metrics
- **Consistent brightness** across frames indicates stable exposure/tracking
- **Sudden drops in brightness** may indicate clouds, dew on optics, or shutter issues
- **Low star counts** may indicate saturation, focus issues, or trailed stars
- **High contrast + many stars** = good frame quality

### Shift Magnitudes
- **Shifts < 2 px**: Normal atmospheric turbulence, tracking errors, or frame-to-frame movement
- **Shifts 2-5 px**: Larger movements, may indicate tracking issues or wind
- **Shifts > 10 px**: Potential mount issues, auto-guiding failures, or manual repositioning
- **Inconsistent patterns**: Systematic drift (linear increase) may indicate tracking bias

## Debugging Common Issues

### Stars Drifting in Stack
1. Check shift magnitudes - are they consistently increasing?
2. Verify reference frame selection (should be well-centered frame)
3. Look for systematic patterns: rotation, linear drift, or random walk
4. Consider re-running with `--no-registration` to see if it's a registration issue

### Frame Rejections
Run with `-v` flag to see:
- Which frames are being rejected
- Their quality metrics
- Why they fall below threshold (with `--quality-filter`)

### Phase Correlation Failures
If shift warnings appear but stacking continues, the fallback centroid method is being used. This is normal for:
- Very noisy or low-contrast images
- Saturated images or those with bright stars
- Images with non-standard artifact patterns

## Technical Details

### SNR Calculation
Signal-to-Noise Ratio is calculated as:
- With photutils: (median - mean) / std_dev
- Fallback: (median - global_mean) / contrast

### Shift Validation
Shifts are accepted if:
1. Phase correlation produces finite values
2. Shift magnitude < 50% of image size
3. Centroid fallback requires ≥10 bright pixels on both images

## Benefits

✅ **Better visibility** into frame quality and rejection reasons
✅ **Early warning** of tracking issues or environmental problems
✅ **More robust registration** resistant to edge cases and noise
✅ **Easier debugging** when troubleshooting drift or quality issues
✅ **Production-ready** diagnostics for automated observation pipelines

## Backward Compatibility

These improvements are fully backward compatible:
- Verbose output is optional (default: silent operation)
- All shift calculations remain unchanged for non-problematic images
- No API changes or required parameter modifications
