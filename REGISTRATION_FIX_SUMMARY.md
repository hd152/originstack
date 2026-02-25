# Registration Fix Summary

## What Was Wrong

Your FITS files were showing **zero shift** `(0.0, 0.0)` for all frames even though you know there's definitely offset. This can happen when:

1. **Phase cross-correlation fails silently** and returns bad values
2. **No fallback method works** due to data characteristics
3. **Registration algorithm can't find enough structure** in the luminance channel
4. **Images have different contrast/structure** making correlation fail

## Solutions Implemented

### 1. **Enhanced Registration Algorithm**
Added three-tier registration method:
- **Phase cross-correlation** (primary, most accurate for well-structured data)
- **Normalized cross-correlation** (NEW: works better on varied contrast images)
- **Centroid fallback** (improved: uses 90th percentile, requires >10 bright pixels)

### 2. **Detailed Debug Output**
Run with `--debug-registration` flag:
```bash
python astro_stack.py -d your_images/ -o output.fits --debug-registration
```

This creates:
- **Console output** showing which method succeeded for each frame
- **PNG images** in `_registration_debug/` showing reference vs each frame
- **Statistics files** with min/max/mean/std for each image

### 3. **Better Error Reporting**
Console now shows:
```
[phase_correlation succeeded: [-0.1 -2.1]]
OR
[phase_correlation rejected: nan/inf or too large]
OR  
[xcorr fallback: shift=(+2.1, -1.5)]
OR
[CRITICAL: no registration method succeeded]
```

### 4. **Reference Statistics**
Shows what the stacker expects:
```
Reference luminance stats - min=29.2, max=713.0, mean=30.4, std=12.5
```

## How to Debug Your Issue

### Step 1: Run with debug flag
```bash
python astro_stack.py -d your_images/ -o output.fits --debug-registration
```

### Step 2: Check the output
- Does it say **`[phase_correlation succeeded]`**?
  - YES → Shift was calculated correctly
  - NO → Registration algorithm failed (see why below)

### Step 3: Examine diagnostic files
Look in `_registration_debug/` folder:
- Open `light_001_ref.png` and `light_001_img.png` side-by-side
- Are the stars offset?
  - YES → Algorithm found real shift (but might report as 0.0 - report this!)
  - NO → Frames might be identical or completely different

### Step 4: Check statistics
```
Reference:
  min=29.16, max=713.02, mean=30.45, std=12.48
Image:
  min=29.15, max=733.74, mean=30.45, std=12.48
```

**Normal values:** std = 10-20 for good astrophotography data
**Problem:** std < 2 = very low contrast, almost uniform

## Common Issues & Fixes

### Issue 1: "Shifts are still all 0.0"
**But PNG images clearly show offset stars**

→ **This is a real bug** - please report with:
- Output from `--debug-registration`
- The PNG image files from `_registration_debug/`
- A description: "Images offset by ~5 pixels but shift reported as (0,0)"

### Issue 2: "Reference std is very low (< 2.0)"
```
Reference luminance stats - ... std=0.5
[CRITICAL: no registration method succeeded]
```

→ **Your reference frame is almost uniform** - might be:
- Blank/overexposed/underexposed frame
- Flat frame mixed with light frames
- Very low SNR or high noise image

**Fix:** Remove bad reference frame or organize by folder

### Issue 3: "One frame has huge shift, others are zero"
```
light_001.fit: shift=(0.0, 0.0) px
light_002.fit: shift=(+2.1, -1.5) px  
light_003.fit: shift=(0.0, +0.0) px ← sudden drop
light_004.fit: shift=(+15.5, +12.2) px ← jump!
```

→ **Guide loss or optical event at frame 4**

**Fix:** Remove frame 4 and re-run, or check guiding logs

### Issue 4: "Stats show images are very different"
```
Reference: min=10, max=900, std=120
Frame 003: min=50, max=200, std=15  ← much lower contrast
```

→ **Frame 003 is different quality/source**

**Fix:** Check if it's a different target, wrong frame type, or settings changed

## Quick Reference

**To debug registration:**
```bash
python astro_stack.py -d images/ -o output.fits --debug-registration
```

**Then check:**
1. Console output - which method succeeded?
2. PNG images - are there obviously offset stars?
3. Statistics - do std values make sense?

**If diagnostics point to data issue:**
→ Use proper folder structure:
```
images/
  lights/      (only light images here)
  darks/       (only dark images here)
  flats/       (only flat images here)
```

**If diagnostics show algorithm failure with good data:**
→ Report with:
- `debug_output.log` (save console output)
- PNG files from `_registration_debug/`
- Brief description of what you see vs what's reported

## Technical Details

The registration system now tries in order:
1. **skimage.registration.phase_cross_correlation** - Most precise, frequency domain
2. **scipy.signal.correlate2d** (normalized) - Better for varied contrast
3. **scipy.ndimage centroid + 90th percentile** - Most robust to noise/artifacts

Each method has safeguards:
- Validates shifts are finite and reasonable
- Requires minimum data (>10 bright pixels)
- Logs exactly why it failed if applicable

## Next Steps

1. **Run `--debug-registration` on your actual data**
2. **Check PNG images** - do they visually show offset?
3. **Compare statistics** - do they match expected ranges?
4. **Report results** if all diagnostics work but shifts still show zero

The debug tools will tell us exactly what's happening so the issue can be fixed!
