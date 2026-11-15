"""Night-fire pair selection: which Aqua x VIIRS night scenes to tile.

The NASA FIRMS archive (public country-year CSVs) is used only to decide which nights and places to tile; its detections
are never an input or a target of a model. Steps:

1. `read_firms`      night detections of both sensors, type 0 (presumed vegetation fire), Aqua only for MODIS.
2. `eligible`        drop 2023 and later, South America, the Sumatra test block (+60 km), Central Kalimantan in
                     September-October 2015 (+150 km) and everything within 60 km of the reference exclusion centres.
3. `candidate_nights` per region the nights with the most eligible Aqua detections ("strong") plus nights where VIIRS
                     sees fire and MODIS (almost) none ("weak"). A night id is the UTC time shifted by -12 h.
4. `night_pairs`     Aqua x S-NPP night pairs of a candidate night from CMR (|dt| <= 30 min, overlap >= 1 km^2), with the
                     number of eligible detections of each sensor inside the overlap.
5. `select_quota`    per region (South-East Asia: every pair with an eligible Aqua detection) pairs are taken year by
                     year round-robin until the region holds about `quota` expected fire-level MODIS pixels
                     (eligible Aqua detections x `rate`), plus up to half as many weak pairs.
"""
from __future__ import annotations

import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import shapely

from .pairs import COLLECTION, MODIS, VIIRS, attach_granules, cmr_granules, find_pairs, granule_geom, granule_record, pair_row
from .splits import BUFFER_KM, INDO_BLOCK, KALIMANTAN_BOX, min_dist_km

FIRMS_URL = "https://firms.modaps.eosdis.nasa.gov/data/country/{sensor}/{year}/{sensor}_{year}_{country}.csv"
FIRMS_SENSOR = {"modis": "aqua", "viirs-snpp": "viirs"}  # archive folder -> tag used in the tables
YEARS = range(2012, 2023)

SEA_COUNTRIES = ["Indonesia", "Brunei_Darussalam", "Cambodia", "Lao_PDR", "Malaysia", "Myanmar", "Papua_New_Guinea",
                 "Philippines", "Singapore", "Thailand", "Timor-Leste", "Vietnam"]
# country -> region (South-East Asian countries are labelled by lon/lat box, see `sea_region`)
COUNTRY_REGION = {
    "Australia": "Australia", "Tanzania": "Tanzania", "India": "South Asia",
    "Angola": "Africa south", "Zambia": "Africa south", "Mozambique": "Africa south",
    "South_Sudan": "Africa north", "Central_African_Republic": "Africa north", "Nigeria": "Africa north", "Ghana": "Africa north",
    "Democratic_Republic_of_the_Congo": "DRC",  # split at the equator, see `region_of_detections`
    "Russian_Federation": "Boreal Eurasia", "Kazakhstan": "Boreal Eurasia",
    "Portugal": "Mediterranean", "Spain": "Mediterranean", "Greece": "Mediterranean",
    "Mexico": "Central America/Mexico", "Guatemala": "Central America/Mexico", "Honduras": "Central America/Mexico", "Nicaragua": "Central America/Mexico",
    "United_States": "North America", "Canada": "North America",
}
COUNTRY_REGION.update({c: "SEA" for c in SEA_COUNTRIES})
SEA_REGIONS = {"Mainland SE Asia", "Borneo", "Papua", "Sulawesi/Maluku", "Java/Nusa Tenggara/Sulawesi S", "Philippines",
               "Malay Peninsula/N Sumatra", "Sumatra west", "Other"}
WEAK_MIN_VIIRS = 30  # a "weak" night has >= this many eligible VIIRS detections and <= 1 eligible Aqua detection
WEAK_PAIR_MIN_VIIRS = 20


# ---------------------------------------------------------------- FIRMS archive
def firms_path(root: Path, sensor: str, year: int, country: str) -> Path:
    return Path(root) / sensor / str(year) / f"{sensor}_{year}_{country}.csv"


def download_firms(root: str | Path, countries: list[str], years=YEARS, sensors=("modis", "viirs-snpp"), log=print) -> list[Path]:
    """Download the FIRMS country-year archive files (public, no key). Existing files are kept; a file the archive does
    not hold (404) is skipped and logged."""
    out = []
    for sensor in sensors:
        for y in years:
            for c in countries:
                f = firms_path(Path(root), sensor, y, c)
                if f.exists() and f.stat().st_size > 0:
                    out.append(f)
                    continue
                f.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with urllib.request.urlopen(FIRMS_URL.format(sensor=sensor, year=y, country=c), timeout=300) as r, open(f.with_suffix(".part"), "wb") as w:
                        while chunk := r.read(1 << 20):
                            w.write(chunk)
                    f.with_suffix(".part").rename(f)
                    out.append(f)
                except urllib.error.HTTPError as e:
                    log(f"skip {sensor} {y} {c}: HTTP {e.code}")
    return out


def read_firms(path: str | Path, sensor: str, country: str) -> pd.DataFrame:
    """Night detections (daynight N, type 0) of one archive file: columns lat, lon, ts (UTC), sensor ('aqua'|'viirs'), country."""
    df = pd.read_csv(path, usecols=["latitude", "longitude", "acq_date", "acq_time", "satellite", "daynight", "type"],
                     dtype={"acq_time": str, "satellite": str, "daynight": str})
    df = df[(df.daynight == "N") & (df.type == 0)]
    if sensor == "aqua":
        df = df[df.satellite == "Aqua"]
    hh = df.acq_time.astype(int)
    df["ts"] = pd.to_datetime(df.acq_date) + pd.to_timedelta(hh // 100 * 60 + hh % 100, unit="m")
    out = df[["latitude", "longitude", "ts"]].rename(columns={"latitude": "lat", "longitude": "lon"})
    out = out.astype({"lat": "float32", "lon": "float32"})
    out["sensor"], out["country"] = sensor, country
    return out


def load_firms(root: str | Path, countries: list[str], years=YEARS) -> pd.DataFrame:
    """All night detections of the archive files found under root (both sensors, the given countries and years)."""
    parts = []
    for folder, tag in FIRMS_SENSOR.items():
        for y in years:
            for c in countries:
                f = firms_path(Path(root), folder, y, c)
                if f.exists():
                    parts.append(read_firms(f, tag, c))
    if not parts:
        raise FileNotFoundError(f"no FIRMS archive files under {root}; run scripts/fetch_firms.py")
    return pd.concat(parts, ignore_index=True).drop_duplicates(["sensor", "lat", "lon", "ts"])


# ---------------------------------------------------------------- regions and eligibility
def sea_region(lat, lon):
    """South-East Asian sub-region of a detection from lon/lat boxes (later boxes win)."""
    r = np.full(len(lat), "Other", object)
    r[(lon >= 108) & (lon <= 119.5) & (lat >= -4.5) & (lat <= 7.5)] = "Borneo"
    r[(lon >= 130) & (lon <= 142) & (lat >= -10.5) & (lat <= 0.5)] = "Papua"
    r[(lon >= 94) & (lon < 100) & (lat >= -6) & (lat <= 6.5)] = "Sumatra west"
    r[(lon >= 100) & (lon < 108) & (lat > 2) & (lat <= 8)] = "Malay Peninsula/N Sumatra"
    r[(lat > 8) & (lon >= 92) & (lon <= 110)] = "Mainland SE Asia"
    r[(lon >= 116) & (lon <= 127) & (lat >= 4) & (lat <= 21)] = "Philippines"
    r[(lon >= 108) & (lon < 130) & (lat < -4.5) & (lat > -11)] = "Java/Nusa Tenggara/Sulawesi S"
    r[(lon >= 118) & (lon < 130) & (lat >= -4.5) & (lat <= 3) & (r == "Other")] = "Sulawesi/Maluku"
    return r


def region_of_detections(df: pd.DataFrame) -> np.ndarray:
    reg = df.country.map(COUNTRY_REGION).to_numpy(object)
    lat, lon = df.lat.to_numpy(np.float64), df.lon.to_numpy(np.float64)
    sea = reg == "SEA"
    if sea.any():
        reg[sea] = sea_region(lat[sea], lon[sea])
    drc = reg == "DRC"
    reg[drc] = np.where(lat[drc] >= 0, "Africa north", "Africa south")
    return reg


def eligibility(df: pd.DataFrame, avoid_centres: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add `why` (first exclusion reason, '' = eligible) and `eligible` to a detection table."""
    lat, lon = df.lat.to_numpy(np.float64), df.lon.to_numpy(np.float64)
    yr, mo = df.ts.dt.year.to_numpy(), df.ts.dt.month.to_numpy()
    why = np.full(len(df), "", object)

    def mark(mask, code):
        m = np.asarray(mask) & (why == "")
        why[m] = code

    b, k, dg = INDO_BLOCK, KALIMANTAN_BOX, 0.6  # 60 km ~ 0.54 deg, applied to lat and lon
    mark(yr >= 2023, "year2023")
    mark((lon < -25) & (lat < 12.5) & (lon > -92), "south_america")
    mark((lon >= b[0] - dg) & (lon <= b[1] + dg) & (lat >= b[2] - dg) & (lat <= b[3] + dg), "sumatra_test_block")
    mark((yr == 2015) & np.isin(mo, (9, 10)) & (lon >= k[0] - 1.5) & (lon <= k[1] + 1.5) & (lat >= k[2] - 1.5) & (lat <= k[3] + 1.5), "kalimantan_sep_oct_2015")
    if avoid_centres is not None and len(avoid_centres):
        ok = why == ""
        dist = np.full(len(df), np.inf)
        dist[ok] = min_dist_km(lat[ok], lon[ok], avoid_centres.lat.to_numpy(), avoid_centres.lon.to_numpy())
        mark(dist < BUFFER_KM, "near_exclusion_centre")
    out = df.copy()
    out["why"], out["eligible"] = why, why == ""
    return out


# ---------------------------------------------------------------- candidate nights and their pairs
def night_id(ts: pd.Series) -> pd.Series:
    """One local night = one id: UTC time shifted by -12 h."""
    return (ts - pd.Timedelta(hours=12)).dt.date


def candidate_nights(det: pd.DataFrame, top_nights: int = 120, top_nights_sea: int = 400, weak_nights: int | None = None,
                     weak_nights_sea: int = 120) -> list[tuple[str, object]]:
    """(region, night id) keys: per region the nights with most eligible Aqua detections (>= 1 in SE Asia, >= 2 elsewhere)
    and the 'weak' nights (VIIRS sees >= 30 detections, Aqua <= 1)."""
    e = det[det.eligible]
    cnt = e.groupby(["region", "nid", "sensor"]).size().unstack(2).fillna(0)
    for c in ("aqua", "viirs"):
        if c not in cnt:
            cnt[c] = 0
    weak_nights = top_nights // 2 if weak_nights is None else weak_nights
    keys = []
    for reg, g in cnt.groupby(level=0):
        sea = reg in SEA_REGIONS
        strong = g[g.aqua >= (1 if sea else 2)].sort_values("aqua", ascending=False)
        keys += list(strong.head(top_nights_sea if sea else top_nights).index)
        weak = g[(g.aqua <= 1) & (g.viirs >= WEAK_MIN_VIIRS)].sort_values("viirs", ascending=False)
        keys += list(weak.head(weak_nights_sea if sea else weak_nights).index)
    return list(dict.fromkeys(keys))


def count_in(det: pd.DataFrame, geom, t0, t1) -> int:
    """Detections of a time-sorted table inside `geom` (lon may be unwrapped > 180) with t0 <= ts < t1."""
    ts = det.ts.to_numpy()
    i, j = np.searchsorted(ts, np.datetime64(t0)), np.searchsorted(ts, np.datetime64(t1))
    d = det.iloc[i:j]
    if d.empty:
        return 0
    lon = d.lon.to_numpy().astype(np.float64)
    if geom.bounds[2] > 180:
        lon = np.where(lon < 0, lon + 360, lon)
    return int(shapely.contains_xy(geom, lon, d.lat.to_numpy().astype(np.float64)).sum())


def night_pairs(region: str, night, g: pd.DataFrame, aqua: pd.DataFrame, viirs: pd.DataFrame, viirs_platform: str = "snpp") -> list[dict]:
    """Aqua x VIIRS night pairs around the detections `g` of one candidate night (CMR, night granules only), each with the number
    of eligible Aqua / VIIRS detections inside its overlap. `aqua`, `viirs`: eligible detections sorted by ts."""
    ses = requests.Session()
    lo, hi = g.ts.min() - pd.Timedelta(minutes=70), g.ts.max() + pd.Timedelta(minutes=70)
    bbox = (max(float(g.lon.min()) - 1, -180), max(float(g.lat.min()) - 1, -90), min(float(g.lon.max()) + 1, 180), min(float(g.lat.max()) + 1, 90))
    a, b = lo.to_pydatetime().replace(tzinfo=timezone.utc), hi.to_pydatetime().replace(tzinfo=timezone.utc)
    pad = timedelta(minutes=35)
    vp = VIIRS[viirs_platform]["fire"]
    mod = [granule_record(x) for x in cmr_granules(COLLECTION[MODIS["fire"]], a, b, bbox, ses)]
    vnp = [granule_record(x) for x in cmr_granules(COLLECTION[vp], a - pad, b + pad, bbox, ses)]
    mod = [r for r in mod if r["url"] and (r["daynight"] or "").upper() == "NIGHT"]
    vnp = [r for r in vnp if r["url"] and (r["daynight"] or "").upper() == "NIGHT"]
    for r in mod + vnp:
        r["geom"] = granule_geom(r["polys"])
    naive = lambda t: pd.Timestamp(t).tz_localize(None)
    rows = []
    for p in find_pairs(mod, {vp: vnp}):
        m, v = p["modis"], p["viirs"]
        rows.append({"region": region, "night": str(night), **pair_row(p),
                     "n_aqua_elig": count_in(aqua, p["overlap"], naive(m["t0"]), naive(m["t1"])),
                     "n_viirs_elig": count_in(viirs, p["overlap"], naive(v["t0"]), naive(v["t1"]))})
    return rows


def candidate_pairs(det: pd.DataFrame, keys, threads: int = 8, viirs_platform: str = "snpp", log=print) -> pd.DataFrame:
    """Pairs of all candidate nights (`night_pairs`), one CMR round per night, sorted by score."""
    e = det[det.eligible]
    aqua = e[e.sensor == "aqua"].sort_values("ts").reset_index(drop=True)
    viirs = e[e.sensor == "viirs"].sort_values("ts").reset_index(drop=True)
    wanted = set(keys)
    groups = {k: g for k, g in e.groupby(["region", "nid"]) if k in wanted}

    def one(item):
        (region, nid), g = item
        try:
            return night_pairs(region, nid, g, aqua, viirs, viirs_platform)
        except requests.RequestException as ex:  # a night whose CMR query keeps failing is skipped, not fatal
            log(f"skip night {region} {nid}: {ex!r}"[:160])
            return []

    rows = []
    with ThreadPoolExecutor(threads) as ex:
        for i, r in enumerate(ex.map(one, groups.items()), 1):
            rows += r
            if i % 100 == 0:
                log(f"{i}/{len(groups)} nights, {len(rows)} pairs")
    df = pd.DataFrame(rows).drop_duplicates(["modis_id", "viirs_id"])
    df["score"] = df.n_aqua_elig + df.n_viirs_elig / 5
    return df.sort_values("score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- selection
def select_quota(cand: pd.DataFrame, existing: pd.DataFrame | None = None, quota: float = 900.0, rate: float = 0.7) -> pd.DataFrame:
    """Choose the pairs to tile from the candidate pairs. Per region other than South-East Asia and the two savanna regions:
    pairs are added year by year (best pair of each year first) until about `quota` expected fire-level MODIS pixels
    (eligible Aqua detections x `rate`, counting what `existing` already holds) are reached. South-East Asia: every
    candidate pair with >= 1 eligible Aqua detection. Each region also gets weak-fire pairs (<= 1 eligible Aqua detection,
    >= 20 eligible VIIRS detections) up to half the number of its strong pairs (SE Asia: half the strong-pair count too)."""
    d = cand.copy()
    d["year"] = pd.to_datetime(d.modis_time, utc=True).dt.year
    have = {}
    if existing is not None and len(existing):
        have = (existing.groupby(existing.region.replace({"Tanzania": "Africa south"})).n_aqua_elig.sum() * rate).to_dict()
    picks = []
    for reg, g in d.groupby("region"):
        if reg in ("Australia", "Tanzania"):
            continue  # extended only when the existing list is thin; not part of this selection
        weak = g[(g.n_aqua_elig <= 1) & (g.n_viirs_elig >= WEAK_PAIR_MIN_VIIRS)].sort_values("n_viirs_elig", ascending=False)
        strong = g[g.n_aqua_elig >= 1].sort_values("score", ascending=False)
        if reg in SEA_REGIONS:
            s = pd.concat([strong.assign(kind="strong"), weak.head(len(strong) // 2).assign(kind="weak")])
        else:
            need, got, chosen = quota - have.get(reg, 0.0), 0.0, []
            by_year = {y: list(x.index) for y, x in strong.groupby("year")}
            while got < need and any(by_year.values()):
                for y in sorted(by_year):
                    if by_year[y] and got < need:
                        i = by_year[y].pop(0)
                        chosen.append(i)
                        got += strong.n_aqua_elig[i] * rate
            s = strong.loc[chosen].sort_values("score", ascending=False)
            s = pd.concat([s.assign(kind="strong"), weak.head(int(round(len(s) / 2))).assign(kind="weak")]).drop_duplicates(["modis_id", "viirs_id"])
        picks.append(s.sort_values("score", ascending=False).assign(rank_in_region=lambda x: np.arange(len(x))))
    sel = pd.concat(picks, ignore_index=True)
    if existing is not None and len(existing):
        key = lambda x: x.modis_id + "|" + x.viirs_id
        sel = sel[~key(sel).isin(set(key(existing)))]
    return sel.sort_values("score", ascending=False).reset_index(drop=True)


def select_pairs(det: pd.DataFrame, existing: pd.DataFrame | None = None, quota: float = 900.0, rate: float = 0.7, threads: int = 8,
                 id_prefix: str = "N", log=print) -> pd.DataFrame:
    """Whole selection from an eligibility-marked detection table (`eligibility`) to a pair table with the six granules of every pair."""
    det = det.copy()
    det["nid"] = night_id(det.ts)
    keys = candidate_nights(det)
    log(f"candidate nights: {len(keys)}")
    cand = candidate_pairs(det, keys, threads, log=log)
    sel = select_quota(cand, existing, quota, rate)
    log(f"selected {len(sel)} of {len(cand)} candidate pairs")
    sel = attach_granules(sel, threads, log=log)
    first = 0
    if existing is not None and len(existing):
        first = 1 + max(int(str(p)[len(id_prefix):]) for p in existing.pair_id)
    sel.insert(0, "pair_id", [f"{id_prefix}{first + i:04d}" for i in range(len(sel))])
    return sel
