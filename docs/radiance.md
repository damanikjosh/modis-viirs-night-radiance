# Thermal (L1B) inputs: brightness temperature

Added in tile schema v2: `{s}_t4`, `{s}_t11` (K) for every pixel of both sensors, from the L1B radiance products
MYD021KM v6.1 (LAADS C1379758607-LAADS), VNP02IMG v2 (C2105091501-LAADS), VJ102IMG v2.1 (C2105083900-LAADS).
L1B granules share the acquisition stamp (`.AYYYYDDD.HHMM.`) and grid of the fire/geolocation granules; `nightrad.data.pairs.attach_granules`
adds their CMR urls to the pair table.

## MODIS (MYD021KM, `EV_1KM_Emissive` [16, 2030, 1354] uint16, bands 20,21,22,23,24,25,27..36)
- radiance `L = (raw - radiance_offsets[i]) * radiance_scales[i]`, W m-2 um-1 sr-1; raw > 32767 is fill/flag
  (65535 fill, 65533 saturated in practice, ...) -> NaN.
- Brightness temperature (Planck, wavelength form): `BT = c2 / (lam * ln(1 + c1 / (lam^5 * L)))`,
  `c1 = 1.191042e8 W um^4 m-2 sr-1`, `c2 = 1.4387752e4 um K`, `lam = 1e4 / nu[cm-1]`.
- Effective central wavenumbers (MODIS L1B documentation): band 21 2505.277 (3.99 um), band 22 2518.028 (3.97 um),
  band 31 908.0884 cm-1 (11.01 um). The small MODIS temperature correction (TCS/TCI) is NOT applied (<~0.5 K).
- `T4` = band 22; where band 22 is flagged/saturated (it saturates at ~331 K) or its BT > 328 K, band 21 (high range, ~500 K)
  is used. `T11` = band 31.

## VIIRS (VNP02IMG/VJ102IMG, `observation_data/I04`, `I05`, [6464, 6400] uint16, 375 m)
- BT from each file's own LUT `I04_brightness_temperature_lut`, `I05_...` (65536 float32): `BT = lut[raw]`.
  raw >= 65528 are flags (65532 Missing_EV, 65533 Bowtie_Deleted, 65534 Cal_Fail, 65535 fill) -> NaN; LUT entries
  <= 0 / non-finite (-999.9 fill) -> NaN.
- **Saturation (audit-21):** the LUT top (361.8 K) is never reached: saturated I4 pixels read 356-358 K in L1B (or fold over to 208-350 K). The fire product marks them (`FP_T4 == 367 K`, `algorithm QA` bit 3); the reader sets `sat = 1` and T4 = 367 K (a lower bound) for those pixels with a finite L1B reading.
- I4 (3.74 um) LUT range 208-361.8 K (so saturated fire pixels read ~362 K, not 367 K in VNP02IMG v2 of this file); I5 (11.45 um) 150-423 K.

## In tiles
Each cell takes the value of its nearest native pixel (same assignment as `pid`). Non-finite -> 0, so **0 = missing**.
The two sensors' 4 um bands differ in wavelength/saturation (MODIS ~3.97 um sat ~331 K -> ~500 K from band 21;
VIIRS I4 3.74 um sat ~362 K), which is part of what the model has to harmonize.
