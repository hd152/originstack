# Registration Debugging Guide

## Problem: Zero Shift Reported

When your stacker reports zero shift `(0.0, 0.0)` for all frames but you know there's definitely offset, use these debugging techniques.

## Quick Diagnostic Mode

Use the `--debug-registration` flag to enable detailed diagnostics:

```bash
python astro_stack.py -d your_images/ -o output.fits --debug-registration
```

This will:
1. Enable verbose output (automatically, via `-v`)
2. Show reference image statistics (min, max, mean, std)
3. Show which registration method succeeded for each frame
4. Create a `_registration_debug/` directory with:
   - PNG visualization of reference vs each frame
   - Statistics files with detailed metrics
   - Help identify contrast, structure, and similarity issues

## Understanding the Output

### Reference Image Statistics
```
Reference luminance stats - min=29.2, max=713.0, mean=30.4, std=12.5
```

**What to look for:**
- **std (standard deviation)** = contrast indicator
  - `std < 5` = very low contrast (almost uniform image)
  - `std 5-20` = normal astrophotography image
  - `std > 20` = very high contrast
- **min/max range** = dynamic range
  - If min ≈ max (e.g., 100, 101), image is essentially uniform
  - Wide range suggests structure (stars, features)

### Per-Frame Diagnostics
```
[phase_correlation succeeded: [-0.1 -2.1]]
[CRITICAL: no registration method succeeded] phase_cc rejected | xcorr rejected | centroid rejected
```

**Possible messages:**
- `phase_correlation succeeded` = Good, precise sub-pixel shift found
- `xcorr fallback: shift=(...)` = Used correlation fallback
- `centroid fallback: shift=(...)` = Used centroid extraction fallback
- `CRITICAL: no registration method succeeded` = All methods failed → investigate why

**Common rejection reasons:**
- `nan/inf or too large` = Phase correlation returned unrealistic value
- `insufficient bright pixels` = Centroid method needs >10 bright pixels above 90th percentile
- `bad result` = Cross-correlation detected invalid shift

### Low Contrast Warning
```
[WARNING: very low contrast] min=100.1, max=101.2, std=0.1 (ref_std=12.5)
```

This means the frame is nearly uniform. Possible causes:
- Heavy overexposure or underexposure
- Cloud cover
- Dew on lens
- Focus issue
- Image is blank/corrupted

## Diagnostic Files

The `_registration_debug/` directory contains:

### PNG Images
- `light_001_ref.png` = Normalized reference image (0-255 greyscale)
- `light_001_img.png` = Normalized comparison image
  
**How to use them:**
1. Open both images side-by-side
2. Look for obvious star position differences
3. Check if images look structurally similar
4. Look for dust marks, optical artifacts
5. If images are completely different = frames are of different objects or very heavy processing

### Stats Files
```
Reference:
  min=29.16, max=713.02, mean=30.45, std=12.48
Image:
  min=29.15, max=733.74, mean=30.45, std=12.48
```

**Compare the stats:**
- If `std` values are very different (e.g., 12.5 vs 2.0), frames have different levels of contrast
- If `min/max` ranges don't overlap, frames might be from different objects
- If both are nearly identical but shift is 0, may be a registration algorithm issue

## Troubleshooting Steps

### 1. First, check that phase_correlation is available
```bash
python -c "from skimage.registration import phase_cross_correlation; print('Available')"
```
If error: install scikit-image
```bash
pip install scikit-image>=0.19
```

### 2. Run debug mode and check output
```bash
python astro_stack.py -d images/ -o output.fits --debug-registration
```

### 3. Read the console output
- Did phase_correlation succeed? → Problem is minor
- Did it fail and fall back? → Check the reason
- Did all methods fail? → Read step 4

### 4. Examine the PNG images
- Are they structurally similar?
  - **YES** → Issue is with registration algorithms (report this!)
  - **NO** → Frames aren't of the same object, only use one frame type
- Are stars obviously offset?
  - **YES** → Debug files prove shift exists, but algorithm isn't finding it
  - **NO** → Offsets are sub-pixel (normal)

### 5. Check the stats files
```bash
cat _registration_debug/light_*.stats.txt
```

Look for:
- **Consistent std values** = good, consistent data
- **Wildly different std** = some frames are different quality
- **Very low std** = nearly uniform, probably not useful frames
- **Very high std** = possible artifacts or saturation

## Common Causes & Solutions

### Case 1: "All shifts are 0.0"
```
[phase_correlation succeeded: [0. 0.]]
```

**Possible causes:**
1. Images are truly identical (exactly aligned by auto-guider) → Normal!
2. Images are rotated slightly (not just translated) → Use `--no-registration`
3. Phase correlation has a bug with your specific data → Use `--debug-registration` and share output

**What to check:**
- Look at PNG images - are stars in same position?
- Check if brightness/contrast are identical→ that's expected for Bayer-pattern raw
- Verify you don't have a mount issue (guide rate = 0?)

### Case 2: "One frame has huge shift"
```
light_003.fit: shift=(+15.2, +8.5) px, magnitude=17.21 px
```

**Causes:**
- Guide star lost during exposure
- Auto-guider reset
- Mount slewed during frame capture
- Optical vibration event

**Solution:**
- Remove that frame and re-run
- Check guiding logs
- Try `--quality-filter` to auto-reject bad frames

### Case 3: "Shifts don't make sense"
```
light_001.fit: shift=(0.0, 0.0) px     
light_002.fit: shift=(+2.1, -1.5) px   
light_003.fit: shift=(-0.1, +0.0) px   
light_004.fit: shift=(+1.8, +15.2) px ← huge jump!
```

**Diagnosis:**
- Run with `--debug-registration`
- Look at light_004_*.png files
- Compare to previous frames - is it different?
- Check stats - is std much lower/higher?

**If stats look normal but PNG looks different:**
→ This is a real bug, please report with `_registration_debug/` files

### Case 4: "Images are definitely different"
```
[CRITICAL: no registration method succeeded]
```

Check the stats:
- Reference: min=10, max=995, std=150
- Frame 3: min=50, max=500, std = 20

**Issue:** Frame 3 has completely different range and contrast
- Maybe it's a frame from different target?
- Maybe it's dark frame or flat frame misclassified?
- Maybe acquisition settings changed (gain, exposure)?

**Solution:**
- Use `process_directory` with proper folder structure:
  ```
  images/
    target_name/
      light/
        *.fits
      dark/
        *.fits
      flat/
        *.fits
  ```

## Advanced: Manual Testing

### Test the phase correlation directly
```python
import numpy as np
from skimage.registration import phase_cross_correlation
from astropy.io import fits

# Load two frames
with fits.open('frame1.fits') as hdu1:
    img1 = hdu1[0].data
with fits.open('frame2.fits') as hdu2:
    img2 = hdu2[0].data

# Convert to luminance if needed
if img1.ndim == 2:
    lum1 = img1
    lum2 = img2
else:
    lum1 = 0.299*img1[:,:,0] + 0.587*img1[:,:,1] + 0.114*img1[:,:,2]
    lum2 = 0.299*img2[:,:,0] + 0.587*img2[:,:,1] + 0.114*img2[:,:,2]

# Test phase correlation
try:
    shift, error, diffphase = phase_cross_correlation(lum1, lum2, upsample_factor=10)
    print(f"Shift: {shift}, Error: {error}, DiffPhase: {diffphase}")
except Exception as e:
    print(f"Error: {e}")
```

### Test centroid method
```python
# Threshold at 90th percentile
thresh = np.percentile(lum1, 90)
mask = lum1 > thresh
print(f"Reference: {mask.sum()} pixels above threshold {thresh:.1f}")

thresh2 = np.percentile(lum2, 90)
mask2 = lum2 > thresh2
print(f"Frame: {mask2.sum()} pixels above threshold {thresh2:.1f}")

# Compute center of mass
from scipy import ndimage
cm1 = ndimage.center_of_mass(lum1 * mask)
cm2 = ndimage.center_of_mass(lum2 * mask2)
print(f"Reference center: {cm1}")
print(f"Frame center: {cm2}")
if cm1 and cm2:
    print(f"Shift: ({cm2[1]-cm1[1]:.1f}, {cm2[0]-cm1[0]:.1f})")
```

## Reporting Issues

If you find a registration problem:

1. **Gather data:**
   ```bash
   python astro_stack.py -d images/ -o output.fits --debug-registration
   ```

2. **Save the output:**
   ```bash
   python astro_stack.py -d images/ -o output.fits --debug-registration > debug_output.log 2>&1
   ```

3. **Provide:**
   - The `debug_output.log` file
   - The entire `_registration_debug/` directory (or at least a few PNG pairs)
   - Brief description: "Shifts reported as 0.0 but images clearly offset" or similar
   - How many frames, image size, and data type (Bayer? RGB? Monochrome?)

4. **Optional:**
   - Include one or two source FITS files if possible
   - Include what you see in the PNG images (e.g., "stars are clearly offset by ~5 pixels")

## Summary Checklist

- [ ] Run with `--debug-registration`
- [ ] Check console output for which method succeeded/failed
- [ ] Look at PNG image pairs visually (are they offset?)
- [ ] Check stats files for consistency
- [ ] Verify debug output matches reality (e.g., std ~10-15 for good astronomy data)
- [ ] If issue persists, gather files for bug report
