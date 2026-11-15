# Harmonized VIIRS fire layer: how to run it

The layer turns Aqua-MODIS night granules into points that show where VIIRS (375 m) would probably have seen fire, for years before VIIRS existed.

```
catalogue.py -> worker.py -> aggregate.py -> points.py
NASA CMR        granule ->    per-granule     monthly cells    monthly display points
(granule list)  model ->      cell files      (de-duplicated)  (monthly-points-viirs/)
                flagged 375 m cells
```

**Fire detections are NOT an input.** The model reads MODIS radiance and geometry (MYD021KM, MYD03). The MYD14 file of the granule is read only for its cloud / water / quality
classes (the same classes the training tiles used); its fire classes are collapsed to "clear land" and never reach the model. The layer finds no fire where MODIS itself saw no
fire-level heat: it adds VIIRS-like 375 m detail around MODIS fire-level pixels. It is an experimental model output, not a NASA product; night only, Aqua only.

## Setup
```
uv sync
```
Checkpoint: `weights/best_val.pt` (1.7 MB). The worker refuses any file whose sha256 is not
`87251b2bb8f151d3e941922862f5f146d7b6e0dff02f519d337ab6e8e523dc26` (`nightrad.inference.version`).

| setting | meaning |
|---|---|
| `NR_DATA_ROOT` | data directory (default `./data`). Downloads go to `$NR_DATA_ROOT/raw/nasa/<PRODUCT>/`, layer outputs to `$NR_DATA_ROOT/layer/` |
| `NR_EARTHDATA_TOKEN` | NASA Earthdata user token (bearer). If unset, `~/.netrc` must hold `machine urs.earthdata.nasa.gov login <user> password <pass>` (chmod 600). Never printed or stored |
| `--root`, `--raw`, `--ckpt`, `--device auto|cuda|cpu` | per-command overrides |

A free Earthdata account (https://urs.earthdata.nasa.gov) with the LAADS DAAC application approved is needed to download MODIS granules. CMR (step 1) needs no login.

## Steps (example: three overpasses, Sumatra, August 2006; about 600 MB download, 1.5 minutes on a laptop CPU)
```
export NR_DATA_ROOT=$PWD/data
uv run python scripts/layer/catalogue.py --bbox 94.5,-11.5,141.5,6.5 --start 2006-08-05 --end 2006-08-06 \
    --ids A2006217.1835,A2006217.1840 --out $NR_DATA_ROOT/cat.jsonl            # 1: granule list (drop --ids for every night granule in the window)
uv run python scripts/layer/worker.py --catalogue $NR_DATA_ROOT/cat.jsonl --bbox 94.5,-11.5,141.5,6.5 --region indonesia   # 2: download + model
uv run python scripts/layer/aggregate.py --catalogue $NR_DATA_ROOT/cat.jsonl                                              # 3: monthly cells + coverage
uv run python scripts/layer/points.py --only-complete                                                                      # 4: display points
uv run python scripts/layer/check_points.py --bbox 94.5,-11.5,141.5,6.5                                                    # independent re-count
```
A granule id is `A{yyyyddd}.{hhmm}` (year, day of year, UTC start time). Re-running a step is safe: the worker skips granules that are already done at the current version
(`--force` redoes them), aggregate and points rebuild their outputs from the files on disk.

### 1. `catalogue.py`
Asks NASA CMR for Aqua MYD021KM / MYD03 / MYD14 (collection 6.1) night granules inside `--bbox W,S,E,N` and `--start/--end`; one JSON line per granule (`id`, `t`, and `l1b`, `geo`, `fm` = file name, https link, size).

### 2. `worker.py`
Per granule: download the three files (`laads.py`; sizes are checked, partial files are never kept), read them (`nightrad.data.modis_l1b_only.read_modis_production`), build 96x96 tiles of 375 m cells
on a fixed lon/lat lattice, run the model on the tiles, apply the documented flag rule (`nightrad.raw.rules.detect`) to the predicted VIIRS-view radiance, and keep each flagged cell once
(owner tile, at least 12 cells from the tile border). Settings that define the product: `--margin 12`, `--tile-rule modis_firelevel` (only tiles within 4.5 km of a MODIS fire-level pixel are run), `--batch 16`.
Outputs in `$NR_DATA_ROOT/layer/`:
- `pixels/year=YYYY/month=MM/<id>.parquet`: one row per flagged cell (only when at least one cell is flagged). Columns: `lon, lat, utc, granule, region, pred4_rad, excess4_rad, bg4_rad, pred_T4_K, pred_dT_K,
  modis4_rad, modis_T4_K, vza, modis_alone_flag, tile_row, tile_col, modis_T11_K, dist_modis_firelevel_km, own_pixel_firelevel, myd14_raw_class`. Header: pipeline version, checkpoint sha256, granule id.
- `_done/<id>.json`: status ok, counts (tiles, valid cells, flagged), sha256 of the three inputs and of the parquet, seconds. `_failed/<id>.json`: error text and attempt counter.
- Exit code 1 when any granule failed.

### 3. `aggregate.py`
Reads the granule files of each month, removes seam duplicates, writes `monthly/YYYY-MM.parquet` (all columns above + `acq_date`), `counts.json`, `coverage.json`, `MANIFEST.json`.
A cell seen twice within 20 minutes (tile seam, consecutive 5-minute granules of one overpass) is one row (smallest view zenith kept); the same cell on another overpass or night is kept.
Coverage compares the granules with a record against `--catalogue` (expected granules): `complete` means every expected granule is ok at the minimum version with no row-count mismatch.

### 4. `points.py`
The display layer, `monthly-points-viirs/YYYY-MM.parquet` (+ `counts.json`, `MANIFEST.json`):
1. cells of one MODIS pixel = same granule, same MODIS 4 um radiance and T4, linked when closer than 1.5 MODIS scan footprints;
2. per pixel keep the hottest k = max(1, round_half_up(0.25 x N)) cells by predicted 4 um excess (N = cells of 375 m in a MODIS pixel at its view zenith; 0.25 = share real VIIRS flags in a MODIS fire-level pixel, median of 83 validation pixels);
3. place the kept cells around the excess-weighted centroid of all flagged cells of the pixel, offset turned by -12 deg (descending Aqua track angle, derived from the orbit, not measured per scene);
4. `scan` / `track` = nominal VIIRS I-band footprint at the cell's view zenith (`viirs_footprint.py`).
Columns: `latitude, longitude, scan, track` (km), `acq_date, confidence ("model"), province (empty), clear (empty), region, granule, pixel_id, rank_in_pixel, n_flagged_in_pixel, n_cells_pixel, k, vza_deg,
excess4_rad, pred_T4_K, pred_dT_K, centroid_lon, centroid_lat, utc`. No free-text columns. `--only-complete` skips months that `coverage.json` does not mark complete.

### Checks
`check_points.py` (rows re-summed from the granule files and records; points per pixel = min(k, flagged); sizes 0.30-0.90 km; dates; no free text), `check_geolocation.py` (cells vs the granule's own MYD03),
`check_pixel_grouping.py` (pixel grouping vs MYD03 pixel centres). Unit tests: `uv run pytest tests/layer` (40 tests, no network, no GPU).

## Version rules
- `PIPELINE_VERSION` (`nightrad/inference/version.py`, now `v1.2`) is written into every record and parquet header. Bump it for every change that can alter the flagged cells (tiling, model call, flag rule, columns).
- aggregate accepts a granule only when its record is ok at a version >= `--min-version` (default: the current version). A record without a version counts as v1.0. Older records are reported as `old_version`; re-run the worker (it redoes them).
- An ok record of a newer version is never overwritten by an older worker; a failed run never replaces an ok record of the same or a newer version.
- History: v1.0 first run; v1.1 out-of-memory retry and size-checked downloads; v1.2 overlapping tiles with interior crop (fixes a tile-edge artefact), tile rule `modis_firelevel`, extra columns, bounding-box clip.

## Reproducibility
- The same inputs, checkpoint, batch size and device give the same bytes. The batch size changes the output slightly (batch composition): the layer uses 16.
- CPU and GPU runs of the same granule agree on the flagged cell set in the end-to-end check below but not bit-wise (predicted radiance differs by up to 1e-3, 0.03 K); earlier measurements found about 0.1 % of flags flipping between CPU and GPU in fire-poor scenes.

## End-to-end check against the stored product (6 Aqua night granules, Indonesia, Aug 2006; worker on a laptop CPU, stored product made on GPU/CPU workers)
| step | result |
|---|---|
| worker, 6 granules | 6 ok, 0 failed; per-granule tiles, valid cells, flagged cells, land tiles, fire-level pixels, dropped orphan flags: identical to the stored records; input sha256 identical |
| cells | 2,509 flagged cells = stored 2,509; the cell sets (lon, lat to 1e-6 deg) are identical in all 5 granules that have flags; radiance columns differ by at most 9e-4 (pred4), 0.026 K (pred_T4_K) |
| aggregate | month 2006-08: complete (6/6 granules), 2,509 cells, 0 seam duplicates in this sample |
| points | 1,380 points = stored 1,380 (per granule 501, 91, 162, 263, 363); pixels per granule identical; k, n_flagged_in_pixel, n_cells_pixel, scan, track, vza equal; positions within 0.29 m; 1:1 match |
| checks | `check_points.py`: 0 failures; `check_geolocation.py` 1835: cells 0.43 km median from the nearest MYD03 pixel centre, 100 % land/coast |
