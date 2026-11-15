"""Split rules for the night-fire tiles: exclusion zones, fire episodes, 100 km spatial blocks, train / val2 assignment, weights.

Held out of every set (a tile is dropped when its centre falls in one):
  * year 2023 and later, South America (`tiles.region_of == 'SA'`),
  * the Sumatra block (100-108 E, 6 S-2 N) and everything within 60 km of it,
  * Central Kalimantan (111-115 E, 3 S-1 N) in September-October 2015 within 150 km (the held-out episode),
  * tiles within 60 km of a reference exclusion centre (`resources/exclusion_centres.csv`, the centres of an earlier held-out set).
Training and held-out (`val2`) tiles are separated by 100 km spatial blocks (`blocks`): tiles closer than 100 km on any date form
one block, a block belongs wholly to one side, so no fire episode (tiles within 100 km and 5 days, `episodes`) is cut, and no train
tile lies within the 60 km buffer of a val2 tile (a 36 km tile has a 51 km diagonal, so 60 km guarantees no footprint overlap).
`val2` is later divided into SELECT and REPORT by `scripts/make_nf2_split.py`.

`split_of` is the build-time label written into every tile by `tiles.py` (the time-block rule of the first dataset); the
finalize step overwrites it with train / val2, so nothing downstream depends on it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HOLDOUT_REGIONS = ("SA",)
INDO_BLOCK = (100.0, 108.0, -6.0, 2.0)  # lon0, lon1, lat0, lat1
KALIMANTAN_BOX = (111.0, 115.0, -3.0, 1.0)  # lon0, lon1, lat0, lat1
KALIMANTAN_BUFFER_KM = 150.0
VAL_YEARS = (2019, 2022)
VAL_MONTHS = (3, 8, 11)
TEST_YEARS = (2023,)
TRAIN_YEARS = (2013, 2016, 2019, 2022)
BUFFER_KM = 60.0  # centre distance
BLOCK_KM = 100.0
EPISODE_KM, EPISODE_DAYS = 100.0, 5
R_KM = 6371.0072
SEA_GROUPS = {"Mainland SE Asia", "Borneo", "Papua", "Sulawesi/Maluku", "Java/Nusa Tenggara/Sulawesi S", "Philippines",
              "Malay Peninsula/N Sumatra", "Sumatra west", "Other"}


def in_indo_block(lat: float, lon: float) -> bool:
    return INDO_BLOCK[0] <= lon <= INDO_BLOCK[1] and INDO_BLOCK[2] <= lat <= INDO_BLOCK[3]


def split_of(lat: float, lon: float, region: str, year: int, month: int) -> tuple[str, str]:
    """Build-time label -> (split, holdout), first match wins: SA -> test; Sumatra block -> test; 2023 -> test;
    2019/2022 in March, August, November -> val; 2013/2016/2019/2022 -> train; any other year -> 'none'."""
    if region in HOLDOUT_REGIONS:
        return "test", region
    if in_indo_block(lat, lon):
        return "test", "indonesia_block"
    if year in TEST_YEARS:
        return "test", "time2023"
    if year in VAL_YEARS and month in VAL_MONTHS:
        return "val", ""
    if year in TRAIN_YEARS:
        return "train", ""
    return "none", ""


def _unit(lat, lon):
    la, lo = np.deg2rad(np.asarray(lat, np.float64)), np.deg2rad(np.asarray(lon, np.float64))
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def _chord(km: float) -> float:
    return 2 * np.sin(km / R_KM / 2)


def min_dist_km(a_lat, a_lon, b_lat, b_lon) -> np.ndarray:
    """For every point of A the great-circle distance (km) to the nearest point of B (inf if B is empty)."""
    from scipy.spatial import cKDTree

    if len(b_lat) == 0:
        return np.full(len(a_lat), np.inf)
    d, _ = cKDTree(_unit(b_lat, b_lon)).query(_unit(a_lat, a_lon))
    return 2 * R_KM * np.arcsin(np.clip(d / 2, 0, 1))


def box_dist_km(lat, lon, box) -> np.ndarray:
    """Approximate distance (km) from points to a lon/lat box (0 inside); flat-earth, 111.19 km per degree."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    dla = np.maximum(np.maximum(box[2] - lat, lat - box[3]), 0) * 111.19
    dlo = np.maximum(np.maximum(box[0] - lon, lon - box[1]), 0) * 111.19 * np.cos(np.deg2rad(lat))
    return np.hypot(dla, dlo)


def exclusion_reason(df: pd.DataFrame, avoid_centres: pd.DataFrame | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(reason, distance) per tile of a table with lat, lon, year, month and `region_cont` (continent code of the tile builder):
    the first exclusion that applies ('' = kept) and the distance (km) to the nearest exclusion centre (inf if not computed)."""
    why = np.full(len(df), "", object)

    def mark(m, code):
        m = np.asarray(m) & (why == "")
        why[m] = code

    mark(df.year >= 2023, "year2023")
    mark(df.region_cont == "SA", "south_america")
    mark(box_dist_km(df.lat, df.lon, INDO_BLOCK) < BUFFER_KM, "sumatra_test_block_60km")
    mark(((df.year == 2015) & df.month.isin([9, 10])) & (box_dist_km(df.lat, df.lon, KALIMANTAN_BOX) < KALIMANTAN_BUFFER_KM), "kalimantan_sep_oct_2015_150km")
    dist = np.full(len(df), np.inf)
    if avoid_centres is not None and len(avoid_centres):
        ok = why == ""
        dist[ok] = min_dist_km(df.lat.to_numpy()[ok], df.lon.to_numpy()[ok], avoid_centres.lat.to_numpy(), avoid_centres.lon.to_numpy())
        mark(dist < BUFFER_KM, "near_exclusion_centre")
    return why, dist


def _components(n: int, pairs: np.ndarray) -> np.ndarray:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else coo_matrix((n, n))
    return connected_components(g, directed=False)[1]


def blocks(lat, lon, link_km: float = BLOCK_KM) -> np.ndarray:
    """Spatial block per tile: connected components of 'centres closer than link_km' (any date). Single linkage, so a block can be
    much larger than link_km."""
    from scipy.spatial import cKDTree

    p = cKDTree(_unit(lat, lon)).query_pairs(_chord(link_km), output_type="ndarray")
    return _components(len(lat), p)


def episodes(df: pd.DataFrame, link_km: float = EPISODE_KM, link_days: int = EPISODE_DAYS) -> np.ndarray:
    """Tile-level fire episodes: connected components of tiles linked when within link_km and link_days (a pair's swath can
    hold several independent episodes). One integer id per row (needs lat, lon, year, month, day)."""
    from scipy.spatial import cKDTree

    day = (pd.to_datetime(dict(year=df.year, month=df.month, day=df.day)) - pd.Timestamp("2000-01-01")).dt.days.to_numpy()
    p = cKDTree(_unit(df.lat.to_numpy(), df.lon.to_numpy())).query_pairs(_chord(link_km), output_type="ndarray")
    p = p[np.abs(day[p[:, 0]] - day[p[:, 1]]) <= link_days] if len(p) else p
    return _components(len(df), p)


def group_of(region) -> np.ndarray:
    """Balance group of a region label: all South-East Asian labels -> 'Southeast Asia', Tanzania -> 'Africa south'."""
    return np.array(["Southeast Asia" if r in SEA_GROUPS else ("Africa south" if r == "Tanzania" else r) for r in region], object)


def choose_split(k: pd.DataFrame, share: float = 0.30, seeds=range(300), cap: float = 1.3, base_seed: int = 20251114):
    """val2 mask over the rows of k (needs lat, lon, region (fine label `nf_region`), group_name, episode, year, modis_fire_px,
    viirs_fire_cells). Blocks go to val2 in random order, per region, while val2 holds at most cap x share of the region's fire
    weight (modis_fire_px + viirs_fire_cells / 5); a region always gets at least one val2 block when it has several. Over `seeds`
    seeds the one is kept whose group shares are closest to 0.2-0.4, ties broken by most val2 episodes and (region, year)
    combinations. Returns (bool mask, info dict)."""
    w = (k.modis_fire_px + k.viirs_fire_cells / 5.0 + 1e-3).to_numpy()
    blk = blocks(k.lat.to_numpy(), k.lon.to_numpy())
    k = k.assign(blk=blk, w=w)
    grp = k.group_name.to_numpy()
    bl = k.groupby("blk").agg(region=("nf_region", lambda s: s.value_counts().index[0]), w=("w", "sum")).reset_index()
    best = None
    for sd in seeds:
        rng = np.random.default_rng(base_seed + sd)
        val = set()
        for reg, g in bl.groupby("region"):
            tot, acc = g.w.sum(), 0.0
            for b in rng.permutation(g.blk.to_numpy()):
                wb = float(g.w[g.blk == b].iloc[0])
                if acc < share * tot and acc + wb <= cap * share * tot:
                    val.add(b)
                    acc += wb
            if not any(b in val for b in g.blk) and len(g) > 1:
                val.add(g.blk.iloc[int(rng.integers(len(g)))])
        iv = k.blk.isin(val).to_numpy()
        shares = {gn: w[iv & (grp == gn)].sum() / max(w[grp == gn].sum(), 1e-9) for gn in set(grp)}
        dev = sum(max(0.0, 0.2 - v, v - 0.4) for v in shares.values())
        nve = int(k.loc[iv, "episode"].nunique())
        nry = int(k[iv].groupby(["nf_region", "year"]).ngroups)
        key = (-round(dev, 3), nve + 0.1 * nry)
        if best is None or key > best[0]:
            best = (key, iv, {"seed": sd, "share_by_group": {a: round(float(b), 3) for a, b in shares.items()}, "val2_episodes": nve,
                              "val2_region_years": nry, "blocks": int(bl.blk.nunique()), "val2_blocks": len(val),
                              "deviation_outside_0.2_0.4": round(dev, 3)})
    return best[1], best[2]


def stratum_weights(split, has_fire, is_random) -> np.ndarray:
    """Per-tile sample weight. Within each split the random tiles (uniform over the overlap) estimate q = share of fire-bearing
    area: q = (random tiles with fire) / (random tiles), pooled over all pairs of the split. Fire-bearing tiles get q * N / N_fire,
    fire-free tiles (1 - q) * N / N_free, so each split's weights average 1 and the weighted fire-bearing share equals q."""
    split, has_fire, is_random = np.asarray(split), np.asarray(has_fire, bool), np.asarray(is_random, bool)
    w = np.ones(len(split), np.float32)
    for sp in np.unique(split):
        m = split == sp
        r = m & is_random
        if r.sum() == 0:
            continue
        q = (has_fire & r).sum() / r.sum()
        n, nf, nz = m.sum(), (m & has_fire).sum(), (m & ~has_fire).sum()
        if nf == 0 or nz == 0:
            continue
        w[m & has_fire] = q * n / nf
        w[m & ~has_fire] = (1 - q) * n / nz
    return w


def region_weights(k: pd.DataFrame, sea_share: float = 0.20, lo: float = 0.02, hi: float = 50.0) -> np.ndarray:
    """Region-balance weight per tile, inside each split: target share / actual share of the group's fire weight
    (modis_fire_px + viirs_fire_cells / 5). Targets: Southeast Asia `sea_share`, every other group an equal part of the rest.
    The sampler weight is sample_weight x region_weight (the tile files keep sample_weight only)."""
    fw = (k.modis_fire_px + k.viirs_fire_cells / 5.0).to_numpy()
    out = np.ones(len(k))
    gn = k.group_name.to_numpy()
    for sp in ("train", "val2"):
        m = (k.split == sp).to_numpy()
        groups = sorted(set(gn[m]))
        if not groups:
            continue
        tot = fw[m].sum()
        tgt = {g: (sea_share if g == "Southeast Asia" else (1 - sea_share) / max(len(groups) - 1, 1)) for g in groups}
        if "Southeast Asia" not in groups:
            tgt = {g: 1 / len(groups) for g in groups}
        for g in groups:
            mm = m & (gn == g)
            act = max(fw[mm].sum() / max(tot, 1e-9), 1e-6)
            out[mm] = float(np.clip(tgt[g] / act, lo, hi))
    return out


def centre_filter(avoid_centres: pd.DataFrame | None, year: int, month: int):
    """Keep-mask function f(lat, lon) -> bool for the fire-centred candidates of a pair acquired in year-month: a candidate is
    kept only where its final tile would survive `exclusion_reason` (so the per-pair tile cap is not spent on tiles that are
    dropped later). Pass to `build_pair_tiles(..., centre_filter=f)`."""
    avoid = (avoid_centres.lat.to_numpy(), avoid_centres.lon.to_numpy()) if avoid_centres is not None and len(avoid_centres) else None

    def f(lat, lon):
        lat, lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
        ok = np.ones(len(lat), bool)
        if avoid is not None:
            ok &= min_dist_km(lat, lon, *avoid) >= BUFFER_KM
        ok &= box_dist_km(lat, lon, INDO_BLOCK) >= BUFFER_KM
        ok &= ~((lon < -25) & (lat < 12.5) & (lon > -92))
        if year == 2015 and month in (9, 10):
            ok &= box_dist_km(lat, lon, KALIMANTAN_BOX) >= KALIMANTAN_BUFFER_KM
        return ok

    return f
