# Product fire-mask -> tile class mapping

Same table for MODIS MYD14 (C6.1, `fire mask` SDS, 1354 wide, 1 km) and VIIRS VNP14IMG / VJ114IMG v002
(`fire mask` variable, 6400 wide, 375 m). Mask values checked on real files (MYD14 values seen: 3,4,5,7,8,9; VNP14IMG likewise).

| product value | meaning (user guides) | tile class |
|---|---|---|
| 0 | not processed, missing input | 0 not observed |
| 1, 2 | not processed (other / obsolete) | 0 |
| 3 | water | 1 water |
| 4 | cloud | 2 cloud |
| 5 | non-fire clear land | 3 clear land, no fire |
| 6 | unknown (pixel could not be classified) | 0 (treated as not observed, excluded from `valid`) |
| 7 | fire, low confidence | 4 fire low |
| 8 | fire, nominal confidence | 5 fire nominal |
| 9 | fire, high confidence | 6 fire high |
| anything else | undefined | 0 |

FRP: `FP_power` (MW) written at (`FP_line`, `FP_sample`) -> `log1p(FRP)` on every tile cell of that pixel (not divided
among cells; all cells of a pixel carry the same value, per the footprint-pooling contract).

## Geometry / other choices
- **Cell -> pixel:** nearest native pixel centre (projected to the tile's LAEA, sphere R=6371007.181 m); the cell is
  assigned only if within half the pixel's diagonal of that centre, else cls 0 / pid -1 (outside swath).
- **Pixel size** (`psize`): central differences of projected pixel-centre spacing, axis 1 (samples) = along-scan,
  axis 0 (lines) = along-track. This is centre spacing, not true footprint.
- **Scan angle:** `sin(scan) = R/(R+h) sin(vza)`, h = 705 km (Aqua) / 824 km (VIIRS); sign negative for samples left
  of swath centre, positive right.
- **VIIRS 375 m geolocation:** VNP03MOD (750 m) upsampled 2x per axis; fine cell k sits at coarse coordinate (k-0.5)/2
  (a 750 m pixel = 2x2 fine pixels), linear interpolation on 3-D unit vectors, linear extrapolation at the ends.
  VNP03IMG would be exact; not used because the card specified VNP03MOD.
- **Time:** granule start from the filename stamp (`.AYYYYDDD.HHMM.`); MODIS 5 min, VIIRS 6 min long; line time is linear
  in the line index. `{s}_time` = mean line time over the tile's pixels, minutes since 00:00 UTC of the MODIS date.
- **overlap** = both sensors have a pixel (geometric), regardless of class. **daynight** = median MODIS SZA > 90.
