# Tile schema (contract between data pipeline and model)

> The current version is **5** (`schema_version`; truthful resampling, reason-coded validity, radiance). `docs/data.md` lists every key of a v5 tile as written by the builder; this file describes v1-v4 and is kept for the meaning of the keys that v5 inherits. Fire-product arrays (`{s}_cls`, `{s}_frp`, `{s}_conf`, raw mask) are stored on disk as `eval_{s}_*` and never reach the model.

**Schema version 4** (stored as `schema_version`; dataset v2 tiles). v2 = v1 plus `{s}_t4`, `{s}_t11`, `has_thermal`, `schema_version`. v3 = v2 plus the keys under *v3 additions* below; all v1/v2 keys keep their meaning, except that `{s}_t4`/`{s}_t11` now have missing values filled (see below). v4 = v3 plus the additive keys under *v4 additions* (nothing removed or renamed; v3 readers work unchanged).

One training sample = one **pair tile**: a square patch on a local equal-area grid where an Aqua MODIS pass and a
VIIRS pass (S-NPP or NOAA-20) both looked within |Δt| ≤ 30 min.

Stored as one compressed `.npz` per tile under `data/tiles/<split>/<tile_id>.npz` (float32 unless noted).

## Grid
- `H = W = 96` cells, cell size 375 m (≈ 36 km square). Grid: local Lambert azimuthal equal-area centred on the tile.
- Arrays are `[C, H, W]` or `[H, W]`, row 0 = north.

## Per sensor `s ∈ {modis, viirs}` (prefix keys with `modis_` / `viirs_`)
| key | shape | meaning |
|---|---|---|
| `{s}_cls` | `[H, W]` uint8 | 0 not observed / outside swath, 1 water, 2 cloud, 3 clear land no fire, 4 fire low, 5 fire nominal, 6 fire high |
| `{s}_frp` | `[H, W]` | log1p(FRP MW) for fire cells, 0 elsewhere |
| `{s}_valid` | `[H, W]` uint8 | 1 where observed ∧ clear ∧ land (cls ∈ {3,4,5,6}) |
| `{s}_scan` | `[H, W]` | sensor scan angle, degrees, signed (−55…55 MODIS, −56…56.6 VIIRS; validated to ±57) |
| `{s}_psize` | `[2, H, W]` | pixel along-scan and along-track size, km |
| `{s}_vza`, `{s}_sza` | `[H, W]` | view and solar zenith, degrees |
| `{s}_pid` | `[H, W]` int32 | native pixel id covering the cell (MODIS: line*1354+sample; VIIRS: line*6400+sample); −1 if none |
| `{s}_time` | scalar | pass time, minutes since tile epoch |
| `{s}_t4`, `{s}_t11` | `[H, W]` | brightness temperature, K, of the cell's native pixel: 4 µm (MODIS band 22, band 21 where 22 is saturated; VIIRS I4) and 11 µm (MODIS band 31; VIIRS I5). **0 = missing** (fill/flag, outside swath, or no L1B); otherwise 150–600 K. Valid cells of both sensors carry a value; fire cells are not special. See `docs/radiance.md` |

## Pair-level
| key | meaning |
|---|---|
| `dt_min` | VIIRS time − MODIS time, minutes |
| `daynight` | 0 day, 1 night |
| `has_thermal` | uint8 scalar: 1 if `_t4/_t11` were filled from L1B for both sensors, else 0 (and they are all 0) |
| `schema_version` | int32 scalar, 2 |
| `overlap` | `[H, W]` uint8: inside both swaths |
| `meta` | JSON string: granule ids, sensor (snpp/noaa20), centre lat/lon, date |

## MODIS footprint pooling
Cells with the same `modis_pid` belong to one MODIS pixel. Model pools fine-cell probabilities over each pid
(soft-OR; uniform weights in v1). The MODIS label for a pid = its `modis_cls` (identical across its cells).

## Loss masks
- detection target valid = `{t}_valid ∧ overlap ∧ (source valid fraction over the target pixel ≥ 0.5)`
- FRP loss only where target cls ∈ {4,5,6}

## v3 additions (schema_version = 3)
All per-sensor `[H, W]` arrays are **constant per native pixel** (all cells with the same `{s}_pid` share the value) and 0 where `{s}_pid == -1`.

| key | dtype | meaning |
|---|---|---|
| `{s}_cloudadj` | uint8 | pixel is not cloud (cls 2) but one of its 8 native neighbours is |
| `{s}_static` | uint8 | pixel centre within max(1 km, half the pixel size) of a FIRMS static-source detection (type 1 volcano, 2 other static land, 3 offshore) in two 5-day windows per dataset year (`scripts/static_sources.py`, dedup to ~10 m). **Class is unchanged**: fire cells on static sources keep cls 4-6 and are only flagged |
| `{s}_t4_missing` | uint8 | T4 or T11 of the pixel was missing (fill, flag, bow-tie deleted). `{s}_t4`/`{s}_t11` then hold the **local median** of the finite observed cells in a 7x7 cell window of the tile, else the tile median, else 0 |
| `{s}_sat` | uint8 | 4 um channel saturated: VIIRS I4 saturated as declared by the fire product (`FP_T4 == 367 K` / QA bit 3; T4 is then set to 367 K); MODIS band 22 flagged/over 328 K so band 21 was used |
| `{s}_t4_bgz`, `{s}_dt_bgz` | float32 | z-score of T4 and of (T4 - T11) against a **21x21 native-pixel** background window around the pixel (computed on the native grid, not the tile grid), using only clear-land (cls 3) pixels that are not static and have finite T4 and T11 (so fire, cloud, water, static and missing pixels are excluded). z = (x - mean) / max(std, 1 K), clipped to +-50. 0 where `bg_ok == 0` |
| `{s}_bg_ok` | uint8 | at least 25% of the 21x21 window is valid background |
| `{s}_conf` | float32 | detection confidence 0-1 on fire cells (cls >= 4), else 0. MODIS: `FP_confidence / 100`; VIIRS: low / nominal / high = 0.3 / 0.6 / 0.9 |
| `random_tile` | uint8 | 1 if the tile centre was drawn uniformly over the VIIRS pixels of the overlap (approximately uniform in area; may contain fire), 0 if centred on a fire pixel |
| `sample_weight` | float32 | see below |
| `region` | str | continent of the tile centre: AF, AS, EU, NA, SA, OC (ordered lon/lat boxes in `tiles.region_of`; coarse: Turkey/Caucasus count as EU/AS by box, Greenland as NA) |

MODIS bow-tie: where scan overlap makes several MODIS pixels cover a cell, the pixel with the smallest view zenith (closest to nadir) among the 3 nearest candidates whose footprint contains the cell centre is kept (`{s}_pid` then points to it).

### Sampling and `sample_weight`
Per pair: up to 40 fire-centred tiles (fire pixels of either sensor, >= 24 km apart, overlap >= 50%) plus up to 20 random tiles. The random tiles estimate the fraction of the overlap that is fire-bearing: `q = (k + 0.5) / (n_random + 1)` with k = random tiles with any fire cell (either sensor). Two strata: fire-bearing tiles (fire-centred and random ones with fire) and fire-free tiles. With `n` tiles in the pair,

    weight(fire-bearing tile) = q * n / n_fire_bearing
    weight(fire-free tile)    = (1 - q) * n / n_fire_free

(inverse selection probability up to a per-pair constant; weights average 1 within a pair, so each pair counts equally). The weighted share of fire-bearing tiles then equals q. All weights are 1 if a pair has no random tile or lacks one stratum. This reproduces the overlap's fire-bearing *tile* rate; the number of fires per fire-bearing tile is still biased up (fire-centred tiles hold more fires than a typical one) - the weight does not correct for that.


## v4 additions (schema_version = 4, dataset v2)
All additive. Builders: `nightrad.data.tiles` (CPU reference) and `tiles_gpu` (torch, CUDA or CPU) produce identical tiles including these keys.

| key | dtype | meaning |
|---|---|---|
| `satellite` | str | VIIRS platform paired with Aqua-MODIS: `snpp` or `noaa20` (`noaa21`) |
| `pair_id` | str | pair id of the v2 pair manifest (`P0000`..`P0799`); empty when a tile is built outside the pipeline |
| `year`, `month`, `day` | int16, int8, int8 | MODIS UTC acquisition date |
| `tile_lat`, `tile_lon` | float64 | tile centre (after the fire offset) |
| `modis_utc`, `viirs_utc` | float64 | mean pass time over the tile, epoch seconds UTC (`dt_min` = (viirs - modis) / 60) |
| `split`, `holdout` | str | split rule verdict (`train`/`val`/`test`; holdout `region`/`time`/empty), `nightrad.data.splits.split_of` |
| `fire_offset_km` | float32 [2] | (east, north) shift of a fire-centred tile centre vs the fire pixel it was seeded from; 0,0 for random tiles |

`daynight` (per tile, from the MODIS solar zenith median) and `{s}_sza` (per cell) are unchanged; cos(SZA) is not stored separately (`cos(deg2rad(sza))`).

Rules new in v4 (builder behaviour, same 96x96 @ 375 m):
- **|dt| filter (D5):** a tile whose own |dt_min| (VIIRS minus MODIS mean line time over the tile) exceeds 30 min is dropped (it does not count toward the 40 fire tiles / 20 random tiles of the pair).
- **Fire offset (D4):** fire-centred tile centres are shifted by U(-12, 12) km east and north in the local tangent plane (`tiles.shift_centres`), so the fire is >= 6 km inside the 36 km tile but not at the centre.
- **`sample_weight`:** the builder still writes the per-pair estimate (see above). `scripts/finalize.py` overwrites it with the global value: within each split, q = (random tiles with any fire cell) / (random tiles), pooled over all pairs, no smoothing; fire-bearing tiles q*N/N_fire, fire-free (1-q)*N/N_free (`splits.stratum_weights`).
- **Splits (D2):** per tile, `split` / `holdout` are the build-time label of `nightrad.data.splits.split_of`; `scripts/finalize.py` overwrites them (train / val + holdout val2) using 100 km spatial blocks and a 60 km buffer (rules in the `nightrad.data.splits` docstring).
- **VIIRS I4 saturation (F1):** `viirs_sat` = product-declared saturation; `viirs_t4` = 367 K there.
- Per-tile manifest columns (not in the npz): `*_fire_px`, `*_fire_px_in`, `*_fire_px_lost` (D3), `viirs_sat_px`, `static_list`.
