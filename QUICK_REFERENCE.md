# Quick Reference: Shift Diagnostics

## One-Line Commands

### View diagnostics with quality metrics
```bash
python astro_stack.py -d your_images/ -o output.fits -v
```

### Stack with quality filtering and diagnostics
```bash
python astro_stack.py -d your_images/ -o output.fits -v --quality-filter
```

## What Each Output Line Means

### Quality Analysis Output
```
light_001.fit: brightness=18.2, contrast=12.1, stars=156
               ↑                ↑               ↑
        Exposure level    Detail/Dynamic     Star detection
                          range indicator    count
```

**Good Frame:** brightness 15-20, contrast 10-15, stars 100+
**Suspect Frame:** brightness <12 or >22, contrast <8, stars <50

### Shift Reporting Output
```
Object_001.fit: shift=(+0.1, +2.1) px, magnitude=2.10 px
               ↑                       ↑
        X and Y offsets         Total shift distance
        (left/right, up/down)   (Pythagoras)
```

**Normal:** magnitude <1 pixel
**Acceptable:** magnitude <5 pixels
**Problem:** magnitude >10 pixels

## Shift Pattern Quick Guide

| Pattern | Magnitude | Fix |
|---------|-----------|-----|
| Stable scatter | <0.5 px | ✅ Perfect |
| Linear increase | 0→3+ px | Check mount tracking |
| Oscillating | 1-2 px repeat | Stabilize mount |
| Sudden jump | jumps 5+ px | Guide star loss → remove frame |
| Circular motion | 0.5-2 px circle | Check leveling/alignment |
| High random | 1-4 px random | Just atmospheric - wait for better seeing |

## Interpret Quality Example

**Real output:**
```
Object_001.fit: brightness=18.2, contrast=12.1, stars=156
Object_002.fit: brightness=18.3, contrast=11.9, stars=163
Object_003.fit: brightness=17.2, contrast= 8.3, stars= 87  ← Problem!
Object_004.fit: brightness=18.1, contrast=12.0, stars=161
Object_005.fit: brightness=18.2, contrast=12.1, stars=159
```

**Analysis:**
- Object 003 has lower brightness (cloud? focus drift? dew?)
- Object 003 has much lower contrast (lack of detail)
- Object 003 detected far fewer stars (trailing? saturation?)
- With `--quality-filter`, this frame will be rejected ✓

**Shifts would tell us:**
```
Object_001.fit: shift=(+0.2, +0.1) px, magnitude=0.22 px
Object_002.fit: shift=(+0.1, +0.2) px, magnitude=0.22 px
Object_003.fit: shift=(+4.2, +5.1) px, magnitude=6.60 px  ← Big shift!
Object_004.fit: shift=(+0.3, +0.0) px, magnitude=0.30 px
Object_005.fit: shift=(+0.1, +0.1) px, magnitude=0.14 px
```

**What it means:**
- Object 003 might be from focus adjustment (explains low quality AND big shift)
- Definitely reject this frame

## Common Questions

**Q: Why do my shifts increase every frame?**
A: Linear drift = tracking bias. Check mount collimation and RA/Dec guiding rates.

**Q: Shifts are all over the place, no pattern?**
A: That's just atmospheric seeing (turbulence). Normal at high magnification.

**Q: One frame has huge shift (10+ px)?**
A: Guide star lost, auto-guider error, or manual correction. Consider removing it.

**Q: Brightness drops suddenly, shift gets bigger?**
A: Likely cloud or focus issue. Quality filter should reject automatically.

**Q: All shifts are >5 px?**
A: Check your guide system calibration. May indicate tracking error or wind.

## Report Issues With This Format

If something looks wrong, copy the full verbose output:
```
python astro_stack.py -d images/ -o output.fits -v --quality-filter 2>&1 | tee diagnostic.log
```

Then share the `diagnostic.log` file. The quality and shift data tells us:
- Frame quality assessment
- Reference frame selection
- Shift calculation results
- Any rejected frames and why

## Read More

- `DIAGNOSTICS_IMPROVEMENTS.md` - Complete feature guide
- `SHIFT_PATTERN_GUIDE.md` - Detailed pattern analysis
- `SESSION_SUMMARY.md` - What was improved and why
- `README.md` - Usage and installation
