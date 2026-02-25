# Shift Pattern Analysis Guide

## Understanding Shift Patterns

The shift diagnostics reported by the stacker reveal important information about your imaging session. Here's how to interpret different patterns.

## Common Shift Patterns

### 1. Stable Tracking (Expected)
```
Object_001.fit: shift=(+0.2, +0.1) px, magnitude=0.22 px
Object_002.fit: shift=(+0.3, +0.0) px, magnitude=0.30 px
Object_003.fit: shift=(+0.1, +0.2) px, magnitude=0.22 px
Object_004.fit: shift=(+0.0, +0.0) px, magnitude=0.00 px
Object_005.fit: shift=(+0.2, +0.1) px, magnitude=0.22 px
Object_006.fit: shift=(+0.1, +0.2) px, magnitude=0.22 px
```
**Characteristics:** Shifts <0.5 px, random scatter, no systematic trend
**Interpretation:** ✅ Excellent tracking, minimal guiding errors
**Action:** No concerns, proceed with stacking

### 2. Linear Drift (Tracking Bias)
```
Object_001.fit: shift=(+0.1, +0.1) px, magnitude=0.14 px
Object_002.fit: shift=(+0.5, +0.3) px, magnitude=0.58 px
Object_003.fit: shift=(+1.2, +0.8) px, magnitude=1.44 px
Object_004.fit: shift=(+2.0, +1.2) px, magnitude=2.33 px
Object_005.fit: shift=(+2.8, +1.5) px, magnitude=3.18 px
Object_006.fit: shift=(+3.5, +2.1) px, magnitude=4.08 px
```
**Characteristics:** Shifts progressively increase, one direction dominates
**Interpretation:** ⚠️ Systematic tracking error or meridian flip issue
**Action:** 
- Check RA/Dec guiding rates
- Verify mount is balanced
- Check for atmospheric refraction effects
- Consider meridian flip timing if near meridian

### 3. Oscillating Pattern (Vibration)
```
Object_001.fit: shift=(-0.3, +1.2) px, magnitude=1.24 px
Object_002.fit: shift=(+0.8, -0.9) px, magnitude=1.21 px
Object_003.fit: shift=(-0.4, +1.1) px, magnitude=1.17 px
Object_004.fit: shift=(+0.7, -0.8) px, magnitude=1.08 px
Object_005.fit: shift=(-0.3, +1.2) px, magnitude=1.24 px
Object_006.fit: shift=(+0.8, -0.9) px, magnitude=1.21 px
```
**Characteristics:** Shifts alternate between +/-, repeating period
**Interpretation:** 🔊 Mount vibration or wind buffeting (periodic)
**Action:**
- Check mount stability and tripod legs
- Investigate wind conditions
- Add mirror vibration damping
- Consider weight distribution

### 4. Sudden Jump (Guidance Loss)
```
Object_001.fit: shift=(+0.2, +0.1) px, magnitude=0.22 px
Object_002.fit: shift=(+0.3, +0.2) px, magnitude=0.36 px
Object_003.fit: shift=(+0.1, +0.0) px, magnitude=0.10 px
Object_004.fit: shift=(+3.5, +4.2) px, magnitude=5.45 px  ← JUMP
Object_005.fit: shift=(+3.8, +4.5) px, magnitude=5.82 px
Object_006.fit: shift=(+3.6, +4.1) px, magnitude=5.38 px
```
**Characteristics:** Sudden large increase in shift that persists
**Interpretation:** 🛰️ Guide star lost, auto-guider reset, or mount slew
**Action:**
- Check guiding logs for guide star loss events
- Verify guide camera focus and exposure
- Check for dew formation
- Consider frame 4+ as separate group for stacking

### 5. Rotational Drift (Field Rotation)
```
Object_001.fit: shift=(+0.1, +0.2) px, magnitude=0.22 px
Object_002.fit: shift=(-0.3, +0.5) px, magnitude=0.58 px
Object_003.fit: shift=(-0.8, +0.3) px, magnitude=0.88 px
Object_004.fit: shift=(-0.9, -0.2) px, magnitude=0.92 px
Object_005.fit: shift=(-0.3, -0.6) px, magnitude=0.67 px
Object_006.fit: shift=(+0.2, -0.5) px, magnitude=0.54 px
```
**Characteristics:** Shifts trace circular/spiral pattern
**Interpretation:** 🔄 Field rotation (common at poles) or leveling issues
**Action:**
- Verify telescope is level
- Check for PA (Position Angle) misalignment
- Consider field rotation in processing (drizzle may help)
- Use rotation-aware stacking if available

### 6. High Variability (Turbulent Conditions)
```
Object_001.fit: shift=(+1.2, +2.1) px, magnitude=2.42 px
Object_002.fit=(-0.8, +3.5) px, magnitude=3.61 px
Object_003.fit: shift=(+2.3, -1.2) px, magnitude=2.60 px
Object_004.fit: shift=(+0.1, +2.8) px, magnitude=2.80 px
Object_005.fit: shift=(-1.5, +0.9) px, magnitude=1.75 px
Object_006.fit: shift=(+1.8, +1.5) px, magnitude=2.34 px
```
**Characteristics:** Shifts 1-3+ pixels, random directions, high variance
**Interpretation:** 🌊 Atmospheric turbulence (seeing conditions poor)
**Action:**
- This is beyond your control - atmospheric effect
- Consider waiting for better seeing
- Higher sub-frame rate helps (shorter exposures)
- Use larger number of frames to average out variations

## Shift Diagnosis Checklist

### Is there a systematic trend?
- **Yes, linear increase:** Check mount tracking and guiding calibration
- **Yes, circular/spiral:** Check leveling, PA alignment, and tube rings
- **No, random scatter:** Normal atmospheric effects

### Are shifts reasonable for exposure duration?
- **Longer exposure:** Expect larger shifts (more time for drift to accumulate)
- **Shorter exposure:** Shifts should be small (<1 pixel typically)
- **Guide interval?:** Shorter guiding intervals reduce drift between captures

### Did something specific change?
```
Frames 1-5: shifts ~0.5 px
Frame 6:    shift jumps to 5.0 px
Frames 7+:  shifts ~5.0 px
```
→ Something happened at frame 6! (guide loss, wind gust, focus adjustment, etc.)

### Quality metric correlation?
```
Low brightness → Higher shifts?
      (clouds → dimmer light → poorer guide performance)

High shifts → Low star count?
      (trailing/guiding loss → fewer detected stars)

Consistent brightness → Variable shifts?
      (atmospheric seeing effect, not focus/exposure)
```

## Quick Reference Table

| Pattern | Magnitude | Cause | Concern | Action |
|---------|-----------|-------|---------|--------|
| Random ±0.2 px | <0.5 px | Normal | None | Proceed |
| Linear increase | 0.1→3+ px | Tracking bias | High | Check calibration |
| Oscillating | 1-2 px periodic | Vibration | Medium | Stabilize mount |
| Sudden jump | 0→5+ px | Guide loss | High | Check guide system |
| Circular pattern | 0.5-2 px | Rotation | Medium | Check PA/level |
| High variance | 1-4 px random | Turbulence | Low | Wait for better seeing |
| Increasing variance | Random pattern | Degrading conditions | Medium | Stop if worsening |

## Advanced Diagnostics: Frame Rejection

### Using Quality Filter
```bash
python astro_stack.py -d images/ -o stack.fits -v --quality-filter
```

This will show:
1. Which frames passed quality analysis
2. Why frames were rejected (if any)
3. The quality threshold applied

### Interpreting Rejections
- **Low brightness:** Image too dim (clouds, moon too high, etc.)
- **Low contrast:** Image lacks contrast (poor focus, excessive haze)
- **Few stars:** Saturation, trailing, or genuine star shortage
- **Below quality score:** Combined effect of all above factors

## Performance Optimization Based on Diagnostics

### If shifts are large but symmetric:
→ Increase crop tolerance to retain more stacked area

### If certain frames show massive shifts:
→ Consider removing outlier frames manually before stacking

### If linear drift is detected:
→ Split imaging session into smaller groups (first 3 frames, next 3 frames)

### If rotational drift is detected:
→ Consider using drizzle stacking (--drizzle-scale 1.5) for better interpolation

## Getting More Detailed Information

Run with maximum verbosity and quality filtering:
```bash
python astro_stack.py -d images/ -o stack.fits -v --quality-filter --verbose
```

This will output:
- Per-frame quality metrics (brightness, contrast, stars, SNR)
- Reference frame selection reasoning
- Shift calculation for each frame
- Frame acceptance/rejection reasons
- Final stacking statistics

## Example Analysis Workflow

1. **Examine shift magnitudes**
   - Are they increasing/decreasing systematically?
   - Are there sudden jumps?
   - Are magnitudes reasonable for exposure time?

2. **Correlate with quality metrics**
   - Do poor shifts correlate with lower quality?
   - Did brightness change?
   - Did star count vary?

3. **Check basic guiding**
   - Are early frames stable?
   - When did shifts become problematic?
   - Any exact timeframes match known events?

4. **Assess data usability**
   - Are shifts <5 pixels → most frames usable
   - Are shifts 5-10 pixels → borderline, check quality
   - Are shifts >10 pixels → likely motion, consider removing

5. **Document for next session**
   - Record seeing conditions
   - Note time of any guidance events
   - Note ambient temperature changes
   - Note wind/vibration issues observed
