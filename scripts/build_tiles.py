"""Build the 96 x 96 tiles (375 m cells) of every pair of a pair table from the downloaded granules.

  python scripts/build_tiles.py --pairs PAIRS.csv [--device cpu|torch-cpu|cuda] [--workers 2] [--download] [--exclusions paper]
                                [--static STATIC.csv] [--only N0454 ...] [--raw DIR] [--out DIR]

Per pair (one directory raw/<pair_id>/<PRODUCT>/<file> from scripts/download.py, or `--download` to fetch, tile, and delete it again):
  tiles/<pair_id>/<pair_id>_NNN.npz   one file per tile, schema in docs/tile_schema.md
  stats/<pair_id>.json                one manifest row per tile (written last: a pair with a stats file is finished and skipped on re-runs)

Devices: `cpu` is the reference implementation (nightrad.data.tiles, numpy / scipy, one process per pair); `cuda` and `torch-cpu` run the same
selection and the same arithmetic in torch (nightrad.data.tiles_gpu), integer and flag layers identical, float layers within 2e-5.
`--exclusions paper` keeps fire-centred candidates only where the final tile survives the exclusion rules of nightrad.data.splits
(the tile cap per pair is then not spent on tiles that finalize would drop); the stored paper tiles were built with it.
Outputs go to $NR_DATA_ROOT/{raw,tiles_build} unless --raw / --out are given."""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")  # one thread per worker process, set before numpy loads

import argparse
import json
import shutil
import sys
import time
import traceback
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from nightrad.data.d3 import count_lost, fire_pixels
from nightrad.data.download import data_root, download_pair, login
from nightrad.data.pairs import pair_paths
from nightrad.data.readers import attach_l1b, read_modis, read_modis_l1b_full, read_viirs, read_viirs_l1b_full
from nightrad.data.splits import centre_filter
from nightrad.data.tables import read_centres, read_pairs
from nightrad.data.tiles import StaticSources, build_pair_tiles, export_names, validate_tile


def tile_rows(pair_id, tiles, fpm, fpv, static_tag=""):
    """One manifest row per tile: build-time split label, acquisition, fire counts per sensor (cells, native pixels, pixels inside
    the tile and pixels that own no cell), validity-class histograms, saturation counts."""
    rows = []
    for n, t in enumerate(tiles):
        la, lo = float(t["tile_lat"]), float(t["tile_lon"])
        row = {"tile": f"{pair_id}_{n:03d}", "pair_id": pair_id, "split": str(t["split"]), "holdout": str(t["holdout"]),
               "region": str(t["region"]), "year": int(t["year"]), "month": int(t["month"]), "day": int(t["day"]),
               "satellite": str(t["satellite"]), "modis_utc": float(t["modis_utc"]), "viirs_utc": float(t["viirs_utc"]),
               "dt_min": float(t["dt_min"]), "daynight": int(t["daynight"]), "random_tile": int(t["random_tile"]),
               "lat": la, "lon": lo, "offset_e_km": float(t["fire_offset_km"][0]), "offset_n_km": float(t["fire_offset_km"][1]),
               "pair_weight": float(t["sample_weight"]), "static_list": static_tag}
        fire = False
        for s, fp in (("modis", fpm), ("viirs", fpv)):
            cls, pid = t[f"{s}_cls"], t[f"{s}_pid"]
            fire |= bool((cls >= 4).any())
            nin, nlost = count_lost(fp, la, lo, pid)
            row.update({f"{s}_fire_cells": int((cls >= 4).sum()), f"{s}_fire_px": int(len(np.unique(pid[cls >= 4]))),
                        f"{s}_fire_px_in": nin, f"{s}_fire_px_lost": nlost, f"{s}_valid_frac": float(t[f"{s}_valid"].mean())})
        for s in ("modis", "viirs"):
            row.update({f"{s}_vc{c}": int(n) for c, n in enumerate(np.bincount(t[f"{s}_vclass"].ravel(), minlength=7))})
            row[f"{s}_dup_cells"] = int((t[f"{s}_ncover"] >= 2).sum())
        row["viirs_sat_px"] = int(len(np.unique(t["viirs_pid"][t["viirs_sat"] > 0]))) if t["viirs_sat"].any() else 0
        row["viirs_sat_fire_px"] = int(len(np.unique(t["viirs_pid"][(t["viirs_sat"] > 0) & (t["viirs_cls"] >= 4)])))
        row["has_fire"] = int(fire)
        rows.append(row)
    return rows


def write_pair(out: Path, pair_id: str, tiles, fpm, fpv, static_tag: str, secs: dict, granule: dict) -> int:
    """Validate and write the tiles of one pair, then its stats file (the finished marker). A tile failing validation
    (e.g. MODIS cells with scan angle > 57 deg: bad edge geolocation) is dropped and logged, not the pair."""
    d = out / "tiles" / pair_id
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    good = []
    for t in tiles:
        t["pair_id"] = np.array(pair_id)
        try:
            validate_tile(t)
            good.append(t)
        except AssertionError:
            with open(out / "invalid_tiles.log", "a") as f:
                f.write(f"{pair_id}\t{float(t['tile_lat']):.3f}\t{float(t['tile_lon']):.3f}\n")
    for n, t in enumerate(good):
        np.savez_compressed(d / f"{pair_id}_{n:03d}.npz", **export_names(t))
    rows = tile_rows(pair_id, good, fpm, fpv, static_tag)
    (out / "stats").mkdir(exist_ok=True)
    tmp = out / "stats" / f"{pair_id}.json.tmp"
    json.dump({"pair_id": pair_id, "rows": rows, "secs": secs, "granule": granule}, open(tmp, "w"))
    os.replace(tmp, out / "stats" / f"{pair_id}.json")
    return len(good)


def _filter(exclusions: str, t0_epoch: float):
    if exclusions == "none":
        return None
    import datetime as dt

    t0 = dt.datetime.fromtimestamp(float(t0_epoch), dt.timezone.utc)
    return centre_filter(read_centres(), t0.year, t0.month)


_STATIC: dict = {}


def _static(path):
    if not path:
        return None
    if path not in _STATIC:
        _STATIC[path] = StaticSources.from_csv(path)
    return _STATIC[path]


def read_pair(job):
    """[worker process] granule files -> MODIS Swath with L1B and the VIIRS raw arrays (torch path) or Swath (cpu path)."""
    raw, sensor, device = job
    t0 = time.time()
    m14, m03, m021, v14, v03, v02 = pair_paths(Path(raw), sensor)
    ms = attach_l1b(read_modis(m14, m03), **read_modis_l1b_full(m021))
    if device == "cpu":
        vs = attach_l1b(read_viirs(v14, v03), **read_viirs_l1b_full(v02))
        return ms, vs, time.time() - t0
    from nightrad.data.readers_gpu import read_viirs_raw

    return ms, read_viirs_raw(v14, v03, v02), time.time() - t0


def build_cpu(job):
    """[worker process] the whole pair on the CPU reference path; returns the tile count."""
    raw, sensor, out, pair_id, opts = job
    ms, vs, t_read = read_pair((raw, sensor, "cpu"))
    t0 = time.time()
    tiles = build_pair_tiles(ms, vs, max_fire_tiles=opts["max_fire_tiles"], static=_static(opts["static"]),
                             centre_filter=_filter(opts["exclusions"], ms.t0))
    fpm, fpv = fire_pixels(ms.lat, ms.lon, ms.cls), fire_pixels(vs.lat, vs.lon, vs.cls)
    return write_pair(Path(out), pair_id, tiles, fpm, fpv, Path(opts["static"] or "").name,
                      {"read": round(t_read, 1), "tiles": round(time.time() - t0, 1)}, {"modis": ms.granule, "viirs": vs.granule})


def build_torch(pair_ids, raws, sensors, out: Path, opts, device: str, workers: int, on_done) -> None:
    """Reader processes (files -> arrays) feed one device loop (all numeric work); writer threads validate and save."""
    import torch
    from nightrad.data.readers_gpu import viirs_gswath
    from nightrad.data.tiles_gpu import GSwath, build_pair_tiles_gpu

    dev = "cpu" if device == "torch-cpu" else device
    static = _static(opts["static"])
    rex, wex = ProcessPoolExecutor(workers), ThreadPoolExecutor(2)
    reads, writes, todo = deque(), deque(), deque(zip(pair_ids, raws, sensors))
    try:
        while todo or reads or writes:
            while todo and len(reads) < workers + 1:
                pid, raw, sensor = todo.popleft()
                reads.append((pid, rex.submit(read_pair, (str(raw), sensor, device))))
            if reads:
                pid, fut = reads.popleft()
                try:
                    ms, vr, t_read = fut.result()
                    t0 = time.time()
                    gm, gv = GSwath(ms, dev), viirs_gswath(vr, dev)
                    tiles = build_pair_tiles_gpu(ms, gv.sw, device=dev, batch=8, max_fire_tiles=opts["max_fire_tiles"], static=static, gm=gm, gv=gv,
                                                 centre_filter=_filter(opts["exclusions"], ms.t0))
                    gi = (gv.cls >= 4).nonzero().cpu().numpy()
                    gl, go = gv.lat.cpu().numpy(), gv.lon.cpu().numpy()
                    fpv = (gl[gi[:, 0], gi[:, 1]].astype(np.float64), go[gi[:, 0], gi[:, 1]].astype(np.float64), (gi[:, 0] * gv.cls.shape[1] + gi[:, 1]).astype(np.int64))
                    ok = np.isfinite(fpv[0]) & np.isfinite(fpv[1])
                    fpv = tuple(x[ok] for x in fpv)
                    fpm = fire_pixels(ms.lat, ms.lon, ms.cls)
                    gran = {"modis": ms.granule, "viirs": gv.sw.granule}
                    del gm, gv
                    if dev.startswith("cuda"):
                        torch.cuda.synchronize(dev)
                    secs = {"read": round(t_read, 1), "tiles": round(time.time() - t0, 1)}
                    writes.append((pid, wex.submit(write_pair, out, pid, tiles, fpm, fpv, Path(opts["static"] or "").name, secs, gran)))
                except Exception:
                    traceback.print_exc()
                    on_done(pid, None)
            while writes and (writes[0][1].done() or len(writes) > 3 or (not todo and not reads)):
                pid, f = writes.popleft()
                try:
                    on_done(pid, f.result())
                except Exception:
                    traceback.print_exc()
                    on_done(pid, None)
    finally:
        rex.shutdown(cancel_futures=True)
        wex.shutdown()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pairs", required=True, help="pair table (csv/parquet); download urls are looked up in CMR when missing and --download is given")
    ap.add_argument("--device", default="cpu", help="cpu (reference), torch-cpu, cuda or cuda:N")
    ap.add_argument("--workers", type=int, default=2, help="processes: one pair each (cpu) or file readers (torch)")
    ap.add_argument("--raw", help="default $NR_DATA_ROOT/raw")
    ap.add_argument("--out", help="default $NR_DATA_ROOT/tiles_build")
    ap.add_argument("--static", help="static-source list from scripts/static_sources.py; without it every static flag is 0")
    ap.add_argument("--exclusions", choices=["none", "paper"], default="none")
    ap.add_argument("--max-fire-tiles", type=int, default=40)
    ap.add_argument("--only", nargs="*", default=[], help="pair ids")
    ap.add_argument("--download", action="store_true", help="fetch the granules of each pair first (Earthdata login) and delete them after tiling")
    ap.add_argument("--keep-raw", action="store_true")
    a = ap.parse_args()
    raw_root = Path(a.raw) if a.raw else data_root() / "raw"
    out = Path(a.out) if a.out else data_root() / "tiles_build"
    out.mkdir(parents=True, exist_ok=True)
    pairs = read_pairs(a.pairs)
    if a.only:
        pairs = pairs[pairs.pair_id.isin(a.only)]
    todo = [r for r in pairs.to_dict("records") if not (out / "stats" / f"{r['pair_id']}.json").exists()]
    print(f"{len(pairs)} pairs, {len(pairs) - len(todo)} already built, {len(todo)} to build on {a.device}", flush=True)
    rows = {r["pair_id"]: r for r in todo}
    if a.download and todo:
        from nightrad.data.pairs import attach_granules

        if "viirs_l1b_url" not in pairs or pairs["viirs_l1b_url"].isna().any():
            rows = {r["pair_id"]: r for r in attach_granules(pairs[pairs.pair_id.isin(rows)]).to_dict("records")}
        session = login()
    ids = [p for p in rows]
    sensors = {p: rows[p].get("viirs_sensor", "snpp") for p in ids}
    opts = {"static": a.static, "max_fire_tiles": a.max_fire_tiles, "exclusions": a.exclusions}
    t_start, stats = time.time(), {"pairs": 0, "tiles": 0, "failed": []}

    def done(pid, n):
        if n is None:
            stats["failed"].append(pid)
        else:
            stats["pairs"] += 1
            stats["tiles"] += n
        print(f"[{time.time() - t_start:6.0f}s] {pid}: " + ("FAILED" if n is None else f"{n} tiles"), flush=True)
        if n is not None and a.download and not a.keep_raw:
            shutil.rmtree(raw_root / pid, ignore_errors=True)

    def run(sub):
        if a.device == "cpu":
            with ProcessPoolExecutor(a.workers) as ex:
                futs = {p: ex.submit(build_cpu, (str(raw_root / p), sensors[p], str(out), p, opts)) for p in sub}
                for p, f in futs.items():
                    try:
                        done(p, f.result())
                    except Exception:
                        traceback.print_exc()
                        done(p, None)
        else:
            build_torch(sub, [raw_root / p for p in sub], [sensors[p] for p in sub], out, opts, a.device, a.workers, done)

    step = max(2 * a.workers, 1) if a.download else max(len(ids), 1)  # --download: fetch, tile and delete a few pairs at a time
    for i in range(0, len(ids), step):
        sub = ids[i:i + step]
        if a.download:
            with ThreadPoolExecutor(4) as ex:
                list(ex.map(lambda p: download_pair(session, rows[p], raw_root), sub))
        run(sub)
    print(json.dumps({**stats, "seconds": round(time.time() - t_start)}))
    sys.exit(1 if stats["failed"] else 0)


if __name__ == "__main__":
    main()
