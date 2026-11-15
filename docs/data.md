# Data pipeline: from NASA granules to training tiles

Aqua MODIS x S-NPP VIIRS scene pairs within 30 minutes -> granules -> night-fire pair selection -> 96 x 96 tiles of 375 m cells -> exclusions, episodes, blocks, train / val2 -> SELECT / REPORT lists -> packs.
Everything is plain files under one root, `NR_DATA_ROOT` (default `./data`); every input is public NASA data.

## Environment

| variable | use |
|---|---|
| `NR_DATA_ROOT` | root of all downloaded and derived files (default `./data`) |
| `NR_EARTHDATA_TOKEN` | Earthdata user token for granule downloads (https://urs.earthdata.nasa.gov); alternative: a `~/.netrc` entry `machine urs.earthdata.nasa.gov login <user> password <password>` (chmod 600). The LAADS DAAC and LP DAAC applications must be approved in the Earthdata profile |
| `FIRMS_MAP_KEY` | free FIRMS map key, only for `scripts/static_sources.py` (https://firms.modaps.eosdis.nasa.gov/api/map_key/) |

Layout below the root: `firms/` (archive csv), `pairs/` (pair tables), `static/` (static-source list), `raw/<pair_id>/<PRODUCT>/` (granules, deleted after tiling with `--download`),
`tiles_build/{tiles,stats}/` (builder output), `tiles/{train,val2}/` + manifests (finalize output), `packs/` (arrays for the model), `episode/` (held-out episode).

## Products (NASA CMR collections; the version strings are read from CMR by `scripts/find_pairs.py --versions`)

| role | Aqua MODIS | S-NPP VIIRS | NOAA-20 VIIRS (optional) | archive |
|---|---|---|---|---|
| calibrated radiance (L1B) | MYD021KM, collection 6.1 (C1379758607-LAADS) | VNP02IMG, version 2 (C2105091501-LAADS) | VJ102IMG, 2.1 (C2105083900-LAADS) | LAADS DAAC |
| geolocation | MYD03, 6.1 (C1379841358-LAADS) | VNP03MOD, 2 (C2105092427-LAADS) | VJ103MOD, 2.1 (C2105084593-LAADS) | LAADS DAAC |
| active fire (cloud / water / glint / bow-tie classes, fire classes kept for evaluation only) | MYD14, 6.1 (C2278858993-LPCLOUD) | VNP14IMG, 2 (C2734202914-LPCLOUD) | VJ114IMG, 2 (C2734197957-LPCLOUD) | LP DAAC |

Verified on 15 Nov 2026 against CMR: MYD021KM 6.1, MYD03 6.1, MYD14 061, VNP02IMG 2, VNP03MOD 2, VNP14IMG 002, VJ102IMG 2.1, VJ103MOD 2.1, VJ114IMG 002.
FIRMS archive (night detections, used only to choose nights and places): `https://firms.modaps.eosdis.nasa.gov/data/country/<modis|viirs-snpp>/<year>/<sensor>_<year>_<Country>.csv`, public, no key.

## Steps

| # | command | reads | writes |
|---|---|---|---|
| 1 | `uv run python scripts/find_pairs.py 2019-10-18 2019-10-18 --night --bbox W S E N` | CMR | `pairs/pairs.csv` |
| 2 | `uv run python scripts/fetch_firms.py [--countries ...] [--years 2012 2022]` | FIRMS archive | `firms/` |
| 3 | `uv run python scripts/select_pairs.py [--existing PAIRS]` | `firms/`, CMR | `pairs/pairs_selected.csv` |
| 4 | `uv run python scripts/download.py PAIRS --only <pair_id>` | CMR, LAADS / LP DAAC | `raw/<pair_id>/<PRODUCT>/<file>` |
| 5 | `FIRMS_MAP_KEY=... uv run python scripts/static_sources.py` | FIRMS area API | `static/static_sources.csv` |
| 6 | `uv run python scripts/build_tiles.py --pairs PAIRS --static STATIC --exclusions paper [--device cuda] [--download]` | `raw/`, static list | `tiles_build/tiles/<pair>/*.npz`, `tiles_build/stats/<pair>.json` |
| 7 | `uv run python scripts/finalize.py --pairs PAIRS` | `tiles_build/` | `tiles/{train,val2}/*.npz`, `manifest.parquet`, `manifest_all.parquet`, `episodes.parquet`, `build_summary.json`, `leak_check.json` |
| 8 | `uv run python scripts/make_nf2_split.py tiles/manifest.parquet configs/nf2_split.json` | manifest | SELECT / REPORT / TRAIN tile lists |
| 9 | `uv run python scripts/pack.py --split train --out packs/train` (and `val2`, `--only-side select`) | `tiles/` | array pack (`meta.json`, one `.npy` per field, `names.json`, `weights.npy`, ...) |
| 10 | `scripts/episode_index.py` -> `scripts/download.py` -> `scripts/build_episode.py` -> `scripts/episode_finalize.py` -> `scripts/episode_pack.py` | CMR, granules | held-out episode tiles and packs (`episode/`) |

`PAIRS` may be the word `paper` (steps 4, 6, 7): the stored list of the 1,017 pairs the paper's tiles were built from (`src/nightrad/data/resources/pairs_paper.csv`: granule ids, times, region, eligible fire counts; download urls are looked up in CMR).
Step 3 re-derives a list from today's FIRMS archive and CMR: similar, not identical (FIRMS and CMR change). Step 2 for all 2012-2022 country files is about 11 GB.

### 1. Pairing (`nightrad.data.pairs`)
No orbit prediction. For every Aqua granule (MYD14 record) the S-NPP granules (VNP14IMG) whose mid-time differs by at most 30 minutes are taken from CMR; footprints (CMR polygons, antimeridian-safe) must overlap by at least 1 km^2.
Fire-product, geolocation and L1B granules share boundaries and acquisition stamp (`.AYYYYDDD.HHMM.`); `attach_granules` finds the other four products of each pair by stamp and drops pairs for which one is missing.
Each 36 km tile is then checked on its own: its time gap is the mean VIIRS scan-line time minus the mean MODIS scan-line time over the tile; tiles above 30 minutes are dropped; a tile must be at least half covered by both swaths.
Terra is not paired.

### 3. Night-fire selection (`nightrad.data.nightfire`)
FIRMS night detections (type 0; Aqua only for MODIS) of 33 countries -> eligible = not 2023 or later, not South America, not the Sumatra block (100-108 E, 6 S-2 N) + 60 km, not Central Kalimantan in September-October 2015 + 150 km, not within 60 km of a reference exclusion centre (`resources/exclusion_centres.csv`: 9,407 centres of an earlier held-out tile set) ->
candidate nights per region (most eligible Aqua detections, plus "weak" nights where VIIRS sees fire and MODIS almost none) -> night pairs from CMR with their eligible detection counts -> per region, years round-robin until about 900 expected fire-level MODIS pixels (SE Asia: every pair with an eligible Aqua detection), plus up to half as many weak pairs.
FIRMS is used only to decide which nights and places to tile; it is never an input or a target.

### 6. Tiles (`nightrad.data.tiles`, `tiles_gpu`, `readers`, `readers_gpu`)
Grid: 96 x 96 cells of 375 m on a local Lambert azimuthal equal-area projection centred on the tile (about 36 km square), row 0 north. Per pair up to 40 fire-centred tiles (fire pixels of either sensor, at least 24 km apart, overlap >= 50 %, centre shifted by U(-12, 12) km east and north) and 20 random tiles (uniform over the VIIRS pixels of the overlap).
Each cell is a bit-exact copy of one native pixel whose footprint contains the cell centre (valid covering pixels first, then the nearest; MODIS: closest to nadir); VIIRS pixels deleted by the bow-tie correction are skipped; nothing is interpolated, blended or filled; radiance, not brightness temperature, is stored (T4, T11 are kept as extra channels).
Validity class per cell and sensor (`{s}_vclass`): 0 off swath, 1 not covered, 2 clear land, 3 cloud, 4 water, 5 bad or missing (including bow-tie-deleted pixels and pixels without a 4 or 11 um value), 6 sun glint (VIIRS); from the products' cloud, water, glint and bow-tie classes, never their fire classes (fire pixels count as clear land). The common mask is `modis_vclass == 2 & viirs_vclass == 2`; VIIRS cells at the 367 K stand-in the fire product injects for saturated pixels are flagged by `viirs_sat` (T4 is set to 367 K there) and are handled downstream (`nightrad.raw`), not by the tiler.
Devices: `cpu` is the reference (numpy / scipy); `cuda` / `torch-cpu` give identical integer and flag layers and float layers within 2e-5 (`docs/gpu_tiles.md`). `--exclusions paper` keeps fire-centred candidates only where the final tile would survive step 7 (the stored tiles were built with it).

Keys of a schema-5 tile (`.npz`, float32 unless noted; `s` = `modis`, `viirs`; arrays are [96, 96] unless noted):

| key | dtype / shape | meaning |
|---|---|---|
| `{s}_vclass` | uint8 | validity class, see above |
| `{s}_valid` | uint8 | 1 where the cell is a valid clear-land observation |
| `{s}_rad4`, `{s}_rad11` | float32 | radiance of the cell's native pixel, 4 um and 11 um bands (MODIS band 22, band 21 where 22 is saturated; VIIRS I4, I5), W m-2 um-1 sr-1; 0 = missing |
| `{s}_t4`, `{s}_t11` | float32 | brightness temperature, K; 0 = missing |
| `{s}_t4_band` | uint8 | MODIS band used for the 4 um value (21 or 22), 0 for VIIRS |
| `{s}_sat` | uint8 | 4 um channel saturated (VIIRS: declared by the fire product, T4 = 367 K) |
| `{s}_t4_missing`, `{s}_bg_ok`, `{s}_cloudadj`, `{s}_static`, `{s}_footprint_complete` | uint8 | quality flags (`docs/tile_schema.md`) |
| `{s}_t4_bgz`, `{s}_dt_bgz` | float32 | z-scores of T4 and T4 - T11 against a 21 x 21 native-pixel background |
| `{s}_pid` | int32 | native pixel id (MODIS line * 1354 + sample; VIIRS line * 6400 + sample), -1 none |
| `{s}_ncover` | uint8 | number of real pixels whose footprint covers the cell |
| `{s}_scan`, `{s}_vza`, `{s}_sza`, `{s}_cossza` | float32 | scan angle, view zenith, solar zenith, cosine of the solar zenith, degrees |
| `{s}_psize` | float32 [2, 96, 96] | pixel size along scan and along track, km |
| `{s}_time`, `{s}_utc` | scalars | pass time (minutes since tile epoch; epoch seconds UTC) |
| `eval_{s}_cls`, `eval_{s}_frp`, `eval_{s}_conf`, `eval_{s}_raw` | uint8 / float32 | the fire product's own classes, power, confidence, raw mask: evaluation only, never inputs or targets |
| `overlap` | uint8 | inside both swaths |
| `dt_min`, `daynight`, `has_thermal`, `schema_version`(=5), `random_tile`, `sample_weight`, `fire_offset_km` [2] | scalars | VIIRS minus MODIS time (min); 1 = night; ... |
| `year`, `month`, `day`, `tile_lat`, `tile_lon`, `satellite`, `pair_id`, `region`, `split`, `holdout`, `meta` | scalars / strings | provenance; `region` = continent code; `split`, `holdout` are overwritten by step 7 |

### 7. Exclusions, episodes, blocks, split (`nightrad.data.splits`, `scripts/finalize.py`)
1. Drop tiles whose centre is in an exclusion zone or within 60 km of an exclusion centre (zones as in step 3, applied to tile centres).
2. Tile-level fire episodes: tiles linked when within 100 km and 5 days.
3. Spatial blocks: tiles closer than 100 km on any date form one block; a block goes whole to train or val2. Per region blocks go to val2 in random order up to about 30 % of the region's fire weight (modis_fire_px + viirs_fire_cells / 5); a seed search (300 seeds) keeps the split whose group shares are closest to 0.2-0.4. No episode is cut.
4. Train tiles within 60 km of a val2 tile are dropped (a 36 km tile has a 51 km diagonal).
5. Weights: `sample_weight` per split = inverse selection probability from the random tiles (a fire-bearing tile gets q N / N_fire, a fire-free tile (1 - q) N / N_free); `region_weight` balances regions (Southeast Asia 20 %, other groups equal).
6. `leak_check.json` recomputes every rule from the final manifest; `finalize.py` exits non-zero unless `ok` is true.

### 8-9. SELECT / REPORT and packs
`scripts/make_nf2_split.py` divides val2 into SELECT and REPORT by 100 km block (`nightrad.raw.nf2_split`), with per-region balance and a fixed seed, and records the leak checks; the lists are in `configs/nf2_split.json`.
`scripts/pack.py` packs one split in manifest order: night tiles whose common mask covers at least 2 % of the tile; adds `eval_episode`, `eval_region` (reporting region), `eval_far_km`; writes `weights.npy` = sample_weight x region_weight.

### 10. Held-out episode (Central Kalimantan, 111-115 E, 3 S-1 N, 14-23 September 2015)
`scripts/episode_index.py` pairs every Aqua x S-NPP overpass of the box (within 35 minutes; the tiler enforces 30 per tile) from CMR; `build_episode.py` tiles a 12 x 12 grid of centres (spacing 37 km) per pair with the same tiler (a tile needs >= 50 % overlap of both swaths and |dt| <= 30 min);
`episode_finalize.py` keeps the tile with most overlap per grid centre and overpass. These tiles carry `split = none`, `holdout = episode-heldout`, and nothing in step 7 reads them.

## Differences between the stored paper tiles and a rebuild
* Tiler code: `tiles.py`, `tiles_gpu.py`, `readers.py`, `readers_gpu.py`, `d3.py` are byte-identical to the versions that built the stored tiles, except one validation line (a tile of a MODIS-only scene may have satellite `none`), which does not touch pair tiles. One code path; the only added option is `centre_filter` on the CPU builder (the torch builder always had it).
* Pair list: stored in `pairs_paper.csv`; a fresh selection (step 3) differs because FIRMS and CMR change.
* Static list: `scripts/static_sources.py` reproduces the stored list exactly (367,737 locations, identical coordinates, types and counts).
* Exclusion centres (60 km): shipped as `resources/exclusion_centres.csv`.

## Verification (15 Nov 2026, Mac CPU, public sources only)
* **Pair tiles.** Pair `N0454` (Aqua x S-NPP night pass of 18 Oct 2019, Java / Nusa Tenggara, 3.0 million km^2 overlap, dt -6.5 min): 6 granules (257 MB) downloaded from LAADS / LP DAAC with `scripts/download.py` (granule ids identical to the stored list), tiled with `scripts/build_tiles.py --exclusions paper` and the stored static list on both the CPU reference (36 tiles, 20 s) and `torch-cpu` (36 tiles, 39 s). The 5 stored training tiles of this pair (kept after finalize) are reproduced: 360 arrays, integer and flag arrays exactly equal, floats within 1e-4. `scripts/finalize.py` and `scripts/pack.py --split train` ran on these tiles and the pack was read back (`nightrad.raw.data.PackedShards`); the split of a single pair is of course not the paper's.
* **Static list.** `scripts/static_sources.py` (FIRMS area API, key from the environment) regenerates the stored list exactly: 367,737 locations, same coordinates, types and counts.
* **Selection.** For the night of 18 Oct 2019 in the Java / Nusa Tenggara region, `nightrad.data.nightfire` (public FIRMS archive, 14 country files of 2019, eligibility with the packaged exclusion centres, CMR pairs) returns the stored pair with its stored eligible counts (10 Aqua, 95 VIIRS detections) and the same granule ids.
* **Held-out episode.** `scripts/episode_index.py` finds 25 Aqua and 33 S-NPP granules in the box (as in the stored index) and 21 pairs, 11 at night. The 11 night pairs were downloaded and tiled on the CPU (`build_episode.py`, `episode_finalize.py`, `episode_pack.py`): 701 tiles built, 134 grid centres dropped for low overlap, 697 night tiles kept after de-duplication (the paper reports 697). The packed arrays `modis_L`, `viirs_L`, `modis_code`, `viirs_code`, `eval_ci`, `eval_cj`, `eval_day_ord` and the tile list equal the stored pack (radiance max difference 0.0).
* Tests: `uv run pytest tests/test_pairs.py tests/test_nightfire.py tests/test_splits.py tests/test_finalize.py tests/test_tiles.py tests/test_tiles_gpu.py tests/test_v2_tiles.py tests/test_truthful_resampling.py tests/test_nf_fixes.py tests/test_region_of.py tests/test_viirs_saturation.py`, one file at a time.
* Not re-run: the full 1,017-pair build and finalize (86 GB of granules, about 17 pairs per minute on a GPU). `finalize.py` and `make_nf2_split.py` were exercised on synthetic and one-pair input only; the stored manifest (`processed/tiles/v3-nightfire2`) was not regenerated.

## Licences and sources
* NASA LAADS DAAC and LP DAAC products (MYD021KM, MYD03, MYD14, VNP02IMG, VNP03MOD, VNP14IMG, VJ1xx): NASA EOSDIS open data, free use with acknowledgement (https://www.earthdata.nasa.gov/engage/open-data-services-software-policies/data-use-guidance).
* NASA FIRMS archive and area API: NASA open data, acknowledgement requested (https://firms.modaps.eosdis.nasa.gov/).
* NASA CMR granule metadata: public (https://cmr.earthdata.nasa.gov/search/).
* Libraries: numpy, scipy, pandas, shapely, h5py, torch (BSD-3), pyarrow, requests (Apache-2.0), pyproj, pyhdf, netCDF4, earthaccess (MIT / BSD-style).
