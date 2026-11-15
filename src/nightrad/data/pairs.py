"""Aqua-MODIS x VIIRS scene pairing from NASA's public granule metadata (CMR, no login, no orbit prediction).

A pair is one Aqua MODIS granule and one VIIRS granule (S-NPP by default, NOAA-20 optional) whose footprints overlap by at
least `min_area_km2` and whose mid-times differ by at most `max_dt_min` minutes. The footprints and times come from the
granule records of the fire products (MYD14, VNP14IMG, VJ114IMG), which share granule boundaries with the Level-1B and
geolocation products; `attach_granules` then finds the other four products of each pair (geolocation and Level-1B).

Products and collections used (CMR concept ids; `collection_versions()` asks CMR for the version strings):

    role      MODIS (Aqua)          VIIRS S-NPP             VIIRS NOAA-20
    L1B       MYD021KM  C61 (6.1)   VNP02IMG   v2           VJ102IMG   v2.1
    geoloc.   MYD03     C61 (6.1)   VNP03MOD   v2           VJ103MOD   v2.1
    fire      MYD14     C61 (6.1)   VNP14IMG   v2           VJ114IMG   v2
"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union

CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"
CMR_COLLECTIONS = "https://cmr.earthdata.nasa.gov/search/collections.json"

# product -> CMR collection concept id
COLLECTION = {
    "MYD021KM": "C1379758607-LAADS", "MYD03": "C1379841358-LAADS", "MYD14": "C2278858993-LPCLOUD",
    "VNP02IMG": "C2105091501-LAADS", "VNP03MOD": "C2105092427-LAADS", "VNP14IMG": "C2734202914-LPCLOUD",
    "VJ102IMG": "C2105083900-LAADS", "VJ103MOD": "C2105084593-LAADS", "VJ114IMG": "C2734197957-LPCLOUD",
}
# VIIRS platform -> product names by role
VIIRS = {
    "snpp": {"l1b": "VNP02IMG", "geo": "VNP03MOD", "fire": "VNP14IMG"},
    "noaa20": {"l1b": "VJ102IMG", "geo": "VJ103MOD", "fire": "VJ114IMG"},
}
MODIS = {"l1b": "MYD021KM", "geo": "MYD03", "fire": "MYD14"}
DATA_EXT = (".hdf", ".nc", ".h5")
STAMP = re.compile(r"\.(A\d{7}\.\d{4})\.")


def _t(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def stamp(granule_id: str) -> str:
    """Acquisition stamp `AYYYYDDD.HHMM` shared by the L1B, geolocation and fire granules of one scan."""
    return STAMP.search(granule_id)[1]


def collection_versions(session: requests.Session | None = None) -> dict[str, str]:
    """product -> version string as CMR reports it (e.g. MYD021KM '6.1', VNP02IMG '2')."""
    s = session or requests
    out = {}
    for prod, cid in COLLECTION.items():
        r = s.get(CMR_COLLECTIONS, params={"concept_id": cid}, timeout=60)
        r.raise_for_status()
        out[prod] = r.json()["feed"]["entry"][0]["version_id"]
    return out


# ---------------------------------------------------------------- CMR
def cmr_granules(collection: str, start: datetime, end: datetime, bbox: tuple | None = None,
                 session: requests.Session | None = None, page_size: int = 2000) -> list[dict]:
    """All CMR granule entries of a collection in [start, end] (optionally inside bbox=(w,s,e,n))."""
    s = session or requests
    params = {
        "collection_concept_id": collection,
        "temporal": f"{start:%Y-%m-%dT%H:%M:%SZ},{end:%Y-%m-%dT%H:%M:%SZ}",
        "page_size": page_size,
        "sort_key": "start_date",
    }
    if bbox:
        params["bounding_box"] = ",".join(str(x) for x in bbox)
    out, page = [], 1
    while True:
        params["page_num"] = page
        for attempt in range(4):
            try:
                r = s.get(CMR, params=params, timeout=120)
                r.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        ent = r.json()["feed"]["entry"]
        out += ent
        if len(ent) < page_size:
            return out
        page += 1


def granule_record(e: dict) -> dict:
    """Reduce a CMR entry to what we need."""
    pid = e.get("producer_granule_id") or e["title"]
    gid = pid
    for ext in DATA_EXT:
        if gid.endswith(ext):
            gid = gid[: -len(ext)]
    url = next((l["href"] for l in e.get("links", [])
                if l.get("rel", "").endswith("/data#") and l["href"].startswith("https")
                and l["href"].endswith(DATA_EXT)), None)
    t0, t1 = _t(e["time_start"]), _t(e["time_end"])
    return {
        "id": gid, "url": url, "t0": t0, "t1": t1, "tmid": t0 + (t1 - t0) / 2,
        "daynight": e.get("day_night_flag"), "size_mb": float(e.get("granule_size", "nan")),
        "polys": e.get("polygons") or [],
    }


def parse_ring(s: str) -> Polygon | None:
    """CMR ring 'lat lon lat lon ...' -> shapely polygon in (lon, lat), antimeridian-unwrapped (lon may be >180)."""
    v = np.array(s.split(), dtype=float)
    if len(v) < 8:
        return None
    lat, lon = v[0::2], v[1::2]
    if lon.max() - lon.min() > 180:  # crosses the antimeridian: unwrap to 0..360
        lon = np.where(lon < 0, lon + 360, lon)
    p = Polygon(np.c_[lon, lat])
    if not p.is_valid:
        p = p.buffer(0)
    return None if p.is_empty else p


def granule_geom(polys: list):
    """Union of outer rings (CMR lists rings as [[outer, hole, ...]]); None if unusable. Lon in -180..360."""
    parts = []
    for ring_group in polys:
        if not ring_group:
            continue
        p = parse_ring(ring_group[0])
        if p is not None:
            parts.append(p)
    return unary_union(parts) if parts else None


def overlap_geom(a, b):
    """Intersection of two granule geometries, trying lon shifts of 360 for antimeridian cases."""
    best = None
    for shift in (0, 360, -360):
        bb = b if shift == 0 else shapely.transform(b, lambda c, s=shift: c + [s, 0])
        if not a.intersects(bb):
            continue
        g = a.intersection(bb)
        if g.is_empty or g.area == 0:
            continue
        if best is None or g.area > best.area:
            best = g
    return best


_GEOD = None


def area_km2(g) -> float:
    global _GEOD
    from pyproj import Geod
    _GEOD = _GEOD or Geod(ellps="WGS84")
    tot = 0.0
    for p in getattr(g, "geoms", [g]):
        if p.geom_type != "Polygon":
            continue
        tot += abs(_GEOD.geometry_area_perimeter(p)[0])
    return tot / 1e6


# ---------------------------------------------------------------- pairing
def find_pairs(modis: list[dict], viirs: dict[str, list[dict]], max_dt_min: float = 30.0,
               min_area_km2: float = 1.0) -> list[dict]:
    """modis: records (granule_record + geom). viirs: fire product name -> records. Returns pair dicts."""
    out = []
    lim = timedelta(minutes=max_dt_min)
    for prod, recs in viirs.items():
        recs = sorted(recs, key=lambda r: r["tmid"])
        tm = np.array([r["tmid"].timestamp() for r in recs])
        for m in modis:
            if m["geom"] is None:
                continue
            lo = np.searchsorted(tm, (m["tmid"] - lim).timestamp(), "left")
            hi = np.searchsorted(tm, (m["tmid"] + lim).timestamp(), "right")
            for v in recs[lo:hi]:
                if v["geom"] is None:
                    continue
                ov = overlap_geom(m["geom"], v["geom"])
                if ov is None:
                    continue
                a = area_km2(ov)
                if a < min_area_km2:
                    continue
                out.append({"modis": m, "viirs": v, "viirs_product": prod, "overlap": ov, "area": a,
                            "dt_min": (v["tmid"] - m["tmid"]).total_seconds() / 60})
    return out


def pair_row(p: dict) -> dict:
    """One pair dict of `find_pairs` -> the flat row stored in a pair table (ids and urls of the two fire granules)."""
    m, v = p["modis"], p["viirs"]
    platform = next(k for k, d in VIIRS.items() if d["fire"] == p["viirs_product"])
    return {"modis_id": m["id"], "modis_url": m["url"], "modis_time": m["tmid"],
            "viirs_id": v["id"], "viirs_url": v["url"], "viirs_time": v["tmid"], "viirs_sensor": platform,
            "dt_min": p["dt_min"], "daynight": (m["daynight"] or "").lower(),
            "overlap_wkt": p["overlap"].wkt, "overlap_km2": p["area"]}


# ---------------------------------------------------------------- the six products of every pair
ROLES = ("modis_fire", "modis_geo", "modis_l1b", "viirs_fire", "viirs_geo", "viirs_l1b")


def _role_products(sensor: str) -> dict[str, str]:
    return {"modis_fire": MODIS["fire"], "modis_geo": MODIS["geo"], "modis_l1b": MODIS["l1b"],
            "viirs_fire": VIIRS[sensor]["fire"], "viirs_geo": VIIRS[sensor]["geo"], "viirs_l1b": VIIRS[sensor]["l1b"]}


def attach_granules(pairs: pd.DataFrame, threads: int = 8, log=print) -> pd.DataFrame:
    """Fill the granule ids and download urls of all six products of every pair from CMR.

    Input needs `modis_id`, `viirs_id` (the two fire granules), `viirs_sensor`, `modis_time`, `viirs_time`. Output adds
    `modis_url`, `viirs_url`, `modis_geo_{id,url}`, `modis_l1b_{id,url}`, `viirs_geo_{id,url}`, `viirs_l1b_{id,url}`.
    Granules are matched by the acquisition stamp of the fire granule (same scan, same boundaries); one CMR query per
    product and UTC day. Pairs for which a product is missing are dropped. If a table already carries granule ids (a
    stored pair list) and CMR now holds a reprocessed granule under the same stamp, the CMR granule is used and the
    count of such changes is logged."""
    df = pairs.copy()
    df["modis_time"] = pd.to_datetime(df["modis_time"], utc=True)
    df["viirs_time"] = pd.to_datetime(df["viirs_time"], utc=True)
    jobs = []
    for sensor in sorted(df["viirs_sensor"].unique()):
        sub = df[df["viirs_sensor"] == sensor]
        prods = _role_products(sensor)
        for side, roles in (("modis_time", ("modis_fire", "modis_geo", "modis_l1b")), ("viirs_time", ("viirs_fire", "viirs_geo", "viirs_l1b"))):
            for _, g in sub.groupby(sub[side].dt.date):
                a = (g[side].min() - pd.Timedelta(minutes=8)).to_pydatetime()
                b = (g[side].max() + pd.Timedelta(minutes=8)).to_pydatetime()
                jobs += [(prods[r], a, b) for r in roles]
    jobs = list(dict.fromkeys(jobs))

    def q(job):
        prod, a, b = job
        return prod, [granule_record(e) for e in cmr_granules(COLLECTION[prod], a, b, None, requests.Session())]

    found: dict[str, dict[str, dict]] = {p: {} for p in COLLECTION}
    with ThreadPoolExecutor(threads) as ex:
        for prod, recs in ex.map(q, jobs):
            for r in recs:
                if r["url"]:
                    found[prod][stamp(r["id"])] = r
    out = {f"{r}_{k}": [] for r in ROLES for k in ("id", "url")}
    keep, changed = [], 0
    for row in df.itertuples():
        st = {"modis": stamp(row.modis_id), "viirs": stamp(row.viirs_id)}
        recs = {r: found[p].get(st[r.split("_")[0]]) for r, p in _role_products(row.viirs_sensor).items()}
        keep.append(all(recs.values()))
        for r, rec in recs.items():
            out[f"{r}_id"].append(rec and rec["id"])
            out[f"{r}_url"].append(rec and rec["url"])
        if recs["modis_fire"] and recs["modis_fire"]["id"] != row.modis_id:
            changed += 1
    for r in ROLES:
        # the two fire granules are named after their sensor: modis_id, viirs_id
        name = {"modis_fire": "modis", "viirs_fire": "viirs"}.get(r, r)
        df[f"{name}_id"] = out[f"{r}_id"]
        df[f"{name}_url"] = out[f"{r}_url"]
    log(f"attach_granules: {sum(keep)} of {len(df)} pairs have all six products; MODIS fire granule re-processed since listing: {changed}")
    return df[keep].reset_index(drop=True)


def product_files(row) -> list[tuple[str, str, str | None]]:
    """(product, granule id, url or None) of the six granules of a pair row (dict or namedtuple-like Series)."""
    g = row.get if hasattr(row, "get") else (lambda k: getattr(row, k))
    v = VIIRS[g("viirs_sensor")]
    return [(MODIS["fire"], g("modis_id"), g("modis_url")), (MODIS["geo"], g("modis_geo_id"), g("modis_geo_url")),
            (MODIS["l1b"], g("modis_l1b_id"), g("modis_l1b_url")),
            (v["fire"], g("viirs_id"), g("viirs_url")), (v["geo"], g("viirs_geo_id"), g("viirs_geo_url")),
            (v["l1b"], g("viirs_l1b_id"), g("viirs_l1b_url"))]


def pair_paths(pair_dir, sensor: str = "snpp"):
    """The six granule files of a downloaded pair directory <pair_dir>/<PRODUCT>/<file>:
    (MYD14, MYD03, MYD021KM, VIIRS fire, VIIRS geolocation, VIIRS L1B)."""
    from pathlib import Path

    one = lambda prod: next((Path(pair_dir) / prod).glob("*.*"))
    v = VIIRS[sensor]
    return one(MODIS["fire"]), one(MODIS["geo"]), one(MODIS["l1b"]), one(v["fire"]), one(v["geo"]), one(v["l1b"])


# ---------------------------------------------------------------- driver
def build_manifest(start: str, end: str, bbox: tuple | None = None, viirs: tuple[str, ...] = ("snpp",),
                   night_only: bool = False, max_dt_min: float = 30.0, log=print) -> pd.DataFrame:
    """Pair table for the UTC dates start..end (YYYY-MM-DD, inclusive): all Aqua x VIIRS pairs within `max_dt_min`,
    with the six granules of each pair (ids, urls)."""
    d0 = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    d1 = datetime.fromisoformat(end).replace(tzinfo=timezone.utc) + timedelta(days=1) - timedelta(seconds=1)
    pad = timedelta(minutes=max_dt_min + 5)
    ses = requests.Session()

    def fetch(prod, a, b):
        recs = [granule_record(e) for e in cmr_granules(COLLECTION[prod], a, b, bbox, ses)]
        recs = [r for r in recs if r["url"] and (not night_only or (r["daynight"] or "").upper() == "NIGHT")]
        for r in recs:
            r["geom"] = granule_geom(r["polys"])
        return recs

    modis = fetch(MODIS["fire"], d0, d1)
    vrecs = {VIIRS[s]["fire"]: fetch(VIIRS[s]["fire"], d0 - pad, d1 + pad) for s in viirs}
    log("CMR granules: " + ", ".join([f"{MODIS['fire']} {len(modis)}"] + [f"{k} {len(v)}" for k, v in vrecs.items()]))
    pairs = find_pairs(modis, vrecs, max_dt_min)
    log(f"pairs |dt|<={max_dt_min:g} min and overlap >= 1 km2: {len(pairs)}")
    if not pairs:
        return pd.DataFrame()
    df = attach_granules(pd.DataFrame([pair_row(p) for p in pairs]), log=log)
    df.insert(0, "pair_id", [f"P{i:05d}" for i in range(len(df))])
    return df
