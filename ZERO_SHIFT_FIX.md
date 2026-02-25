# Zero-Shift Phase Correlation Fix

## Your Symptom

Phase correlation is reporting `[0. 0.]` for all frames, but you know there's definitely offset:

```
Registration: calculating shifts for 52 frames
  ...
  [phase_correlation succeeded: [0. 0.]]
  Light0001.fits: shift=(+0.0, +0.0) px
  [phase_correlation succeeded: [0. 0.]]
  Light0002.fits: shift=(+0.0, +0.0) px
```

All 52 frames showing identical **zero shift** = **algorithm failure, not perfect tracking**

## What Fixed in This Release

### 1. **Normalized Input to Phase Correlation** ✅
Phase correlation now normalizes images (zero mean, unit variance) before running. This is critical for proper operation with 14-bit FITS data.

### 2. **Improved Cross-Correlation Fallback** ✅
Added normalized cross-correlation with peak strength checking before accepting results.

### 3. **Multiple Percentile Thresholds for Centroid** ✅
Centroid fallback now tries percentiles 95, 90, 85, 80 to find one that works with your specific data.

### 4. **Zero-Shift Detection and Suggestion** ✅
If >80% of frames show zero shift, the tool now warns and suggests:
```
python astro_stack.py --skip-phase-correlation ...
```

### 5. **Skip Phase Correlation Flag** ✅
New `--skip-phase-correlation` flag forces use of only fallback methods (xcorr + centroid).

## How to Diagnose Your Issue

### Step 1: Test with Fallback Methods
```bash
python astro_stack.py -d your_images/ -o output.fits -v --skip-phase-correlation
```

**Check the output:**
- Does it show `[xcorr fallback: shift=...]` with non-zero values?
  - **YES** → Phase correlation is the problem (see fix below)
  - **NO** → Fallback methods also failing (see alternative approach below)

### Step 2: If Fallback Methods Work
```bash
python astro_stack.py -d your_images/ -o output.fits -v --skip-phase-correlation --debug-registration
```

This will show PNG diagnostics. If they look correct, then **phase correlation has a bug with your data**.

## Possible Root Causes

### Cause 1: Phase Correlation Bug with 14-bit Data
Your reference stats show:
```
min=5262.3, max=16383.8, std=1104.3
```

This is 14-bit data (max ≈ 2^14). Phase correlation might have numerical issues with this range.

**Test:** Compare with synthetic 8-bit data:
```bash
# Create 8-bit synthetic test
python tools/create_synthetic.py
python astro_stack.py -d synthetic_data -o test.fits -v
```

If synthetic works but your data doesn't → phase correlation has issue with 14-bit

### Cause 2: All Images Are Actually Identical
Less likely but possible if:
- Same frame duplicated?
- Saved twice?
- No movement during capture?

**Check:** Use `--debug-registration` and visually inspect the PNG images

### Cause 3: Images Are Rotated (Not Just Translated)
Phase correlation works on translation only, not rotation.

**Check:** Visual inspection of PNG images - are stars rotated?

## Solutions Available

### Solution 1: Use Fallback Methods (Recommended)
```bash
python astro_stack.py -d your_images/ -o output.fits --skip-phase-correlation -v
```

Fallback methods should work if:
- Images have enough stars (>10 bright pixels)
- Images aren't severely degraded
- Offset is less than 50% of image size

### Solution 2: Improve Phase Correlation Input
The normalization I added should help. Try standard run first:
```bash
python astro_stack.py -d your_images/ -o output.fits -v
```

If still getting zeros, then use Solution 1.

### Solution 3: Use Debug Mode to Understand
```bash
python astro_stack.py -d your_images/ -o output.fits --debug-registration
```

Look in `_registration_debug/` folder:
1. Check if PNG images show visible offset
2. Check stats files for contrast differences
3. If PNG shows offset but shift is (0,0) → real bug for reporting

### Solution 4: Manual Registration
If all else fails, you can skip registration:
```bash
python astro_stack.py -d your_images/ -o output.fits --no-registration
```

This treats all frames as already aligned. Only use if you're sure they're already registered.

## What to Try in Order

**Priority 1 - Normalization Fix**
```bash
python astro_stack.py -d your_images/ -o output.fits -v
```
This now has improved normalization. If it works, problem solved!

**Priority 2 - Fallback Methods**
```bash
python astro_stack.py -d your_images/ -o output.fits -v --skip-phase-correlation
```
If this shows shifts → use it instead of phase correlation

**Priority 3 - Debug & Diagnose**
```bash
python astro_stack.py -d your_images/ -o output.fits --debug-registration
```
Check PNG images and decide next step based on visuals

**Priority 4 - Report If Broken**
If Phase Correlation is broken but debug shows offset:
- Save: `debug_output.log` (full console output)
- Save: `_registration_debug/` folder (PNG + stats)
- Note: How many pixels offset? (estimate from PNG)
- Report: "Phase correlation reports [0,0] but PNG shows ~X pixel offset"

## Testing the Fix

### For Synthetic Data (Should Work)
```bash
python astro_stack.py -d synthetic_data -o test.fits -v
```
Expected: Shows detected shifts like `(-2.1, -0.1)`, `(+4.1, -1.9)`, etc.

### For Your Real Data (Testing)
```bash
# First, test fallback methods
python astro_stack.py -d your_images/ -o test_fallback.fits -v --skip-phase-correlation

# If that works, try normal (with normalization fix)
python astro_stack.py -d your_images/ -o test_normal.fits -v

# If normal still shows zeros, stick with fallback
python astro_stack.py -d your_images/ -o final_output.fits --skip-phase-correlation
```

## Expected Behavior After Fix

### Good Case (Shifts Detected)
```
[phase_correlation succeeded: [-2.1 -0.1]]
Light0001.fits: shift=(-0.1, -2.1) px, magnitude=2.10 px
⬆️ Non-zero shift detected
```

### Fallback Case (Phase CC Failed, Fallback Worked)
```
[xcorr fallback: shift=(+1.5, -2.2)]
Light0002.fits: shift=(-2.2, +1.5) px, magnitude=2.67 px
⬆️ Fallback method found offset
```

### Problem Case (All Methods Failed)
```
[CRITICAL: no registration method succeeded]
Light0003.fits: shift=(+0.0, +0.0) px, magnitude=0.00 px
⬆️ Investigate with --debug-registration
```

### Suspicious Case (Detect & Warn)
```
[WARNING] 48/52 frames have zero shift - this is suspicious!
[SUGGESTION] Try running with --skip-phase-correlation ...
```

## Technical Details: What Changed

### Before
- Phase correlation: raw input data
- No fallback peak strength check
- Only tried 90th percentile for centroid
- Silent failure when all zeros returned

### After
- Phase correlation: **normalized input** (zero mean, unit variance)
- Cross-correlation: checks peak strength vs background
- Centroid: tries 4 different percentiles (95, 90, 85, 80)
- **Warns when suspicious all-zero pattern detected**
- **Can skip phase correlation** with flag
- Better debug output showing which method worked

## Impact

**Your Data:**
- If normalization fixes it → no change needed, just update code (✅ fixed)
- If fallback works → use `--skip-phase-correlation` flag (✅ available)
- If whole thing broken → debug data available for fixing (✅ diagnostic tools ready)

**Synthetic/Library Data:**
- Should continue working as before
- May see improved performance on edge cases
- All tests pass ✅
