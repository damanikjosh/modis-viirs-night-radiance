"""Episode tiles: truthful tiles over the Kalimantan box (111-115 E, 3 S-1 N), every Aqua x S-NPP overpass, on a regular grid of tile centres.

  python scripts/build_episode.py --pairs PAIRS.csv [--raw DIR] [--out DIR] [--device cpu|cuda] [--workers 2] [--download] [--only ID ...]

PAIRS.csv from scripts/episode_index.py. Same tiler as the training tiles (nightrad.data.tiles_gpu.build_tiles_gpu; `cpu` = torch on CPU), 12 x 12 grid of
36 km tiles (centre spacing 37 km); a tile is kept when both swaths cover >= 50 % of it and |dt| <= 30 min. No fire-product information reaches an input
channel; the product's fire / cloud / land classes are kept only under the `eval_*` names. Tiles carry split 'none', holdout 'episode-heldout'.
Output: <out>/tiles/<pair>/<pair>_<ci>_<cj>.npz and <out>/stats/<pair>.json (written last; a pair with a stats file is skipped).
Defaults: raw = $NR_DATA_ROOT/raw, out = $NR_DATA_ROOT/episode."""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "2")
import argparse
import json
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from nightrad.data.download import data_root, download_pair, login
from nightrad.data.pairs import pair_paths
from nightrad.data.tables import read_pairs

BOX = (111.0, 115.0, -3.0, 1.0)  # lon_min lon_max lat_min lat_max
GRID_N = 12


def centres():
    """(ci, cj, lat, lon) of the grid; ci = 0 is the SOUTH row (latitude ascending), cj = 0 the west column."""
    la = BOX[2] + (np.arange(GRID_N) + 0.5) * (BOX[3] - BOX[2]) / GRID_N
    lo = BOX[0] + (np.arange(GRID_N) + 0.5) * (BOX[1] - BOX[0]) / GRID_N
    return [(i, j, float(la[i]), float(lo[j])) for i in range(GRID_N - 1, -1, -1) for j in range(GRID_N)]


def run_job(pair_id: str, raw: str, out: str, device: str):
    import torch
    from nightrad.data.readers import attach_l1b, read_modis, read_modis_l1b_full
    from nightrad.data.readers_gpu import read_viirs_raw, viirs_gswath
    from nightrad.data.tiles import export_names, validate_tile
    from nightrad.data.tiles_gpu import GSwath, build_tiles_gpu

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    out = Path(out)
    if (out / "stats" / f"{pair_id}.json").exists():
        return pair_id, "skip", 0
    t0 = time.time()
    m14, m03, m021, v14, v03, v02 = pair_paths(Path(raw) / pair_id)
    ms = attach_l1b(read_modis(m14, m03), **read_modis_l1b_full(m021))
    dev = "cpu" if device == "cpu" else device
    gm, gv = GSwath(ms, dev), viirs_gswath(read_viirs_raw(v14, v03, v02), dev)
    t_read = time.time() - t0
    cs = centres()
    lat, lon = [c[2] for c in cs], [c[3] for c in cs]
    todo = [c for c, k in zip(cs, gm.covered(lat, lon, km=40.0) & gv.covered(lat, lon, km=40.0)) if k]
    d = out / "tiles" / pair_id
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    rows, dropped = [], {"low_overlap": 0, "dt": 0, "invalid": 0}
    for b in range(0, len(todo), 8):
        batch = todo[b:b + 8]
        for c, t in zip(batch, build_tiles_gpu(gm, gv, [(x[2], x[3]) for x in batch], None, [False] * len(batch))):
            if float(t["overlap"].mean()) < 0.5:
                dropped["low_overlap"] += 1
                continue
            if abs(float(t["dt_min"])) > 30.0:
                dropped["dt"] += 1
                continue
            t["has_viirs"] = np.uint8(1)
            t["split"], t["holdout"] = np.array("none"), np.array("episode-heldout")
            t["pair_id"], t["episode"] = np.array(pair_id), np.array("kalimantan-2015")
            try:
                validate_tile(t)
            except AssertionError as e:
                dropped["invalid"] += 1
                open(out / "invalid_tiles.log", "a").write(f"{pair_id}\t{c[0]}_{c[1]}\t{e}\n")
                continue
            name = f"{pair_id}_{c[0]:02d}_{c[1]:02d}"
            np.savez_compressed(d / (name + ".npz"), **export_names(t))
            row = {"tile": name, "job": pair_id, "episode": "kalimantan-2015", "ci": c[0], "cj": c[1], "lat": c[2], "lon": c[3], "dn": int(t["daynight"]),
                   "date": f"{int(t['year']):04d}-{int(t['month']):02d}-{int(t['day']):02d}", "has_viirs": 1,
                   "modis_utc": float(t["modis_utc"]), "viirs_utc": float(t["viirs_utc"]), "dt_min": float(t["dt_min"]), "overlap_frac": float(t["overlap"].mean())}
            for s in ("modis", "viirs"):
                row.update({f"{s}_vc{k}": int(v) for k, v in enumerate(np.bincount(t[f"{s}_vclass"].ravel(), minlength=7))})
                row[f"{s}_valid_frac"] = float(t[f"{s}_valid"].mean())
                row[f"{s}_fire_cells"] = int((t[f"{s}_cls"] >= 4).sum())
            row["common_valid_frac"] = float(((t["modis_vclass"] == 2) & (t["viirs_vclass"] == 2)).mean())
            rows.append(row)
    (out / "stats").mkdir(exist_ok=True)
    json.dump({"job": pair_id, "rows": rows, "dropped": dropped, "secs": {"read": round(t_read, 1), "build": round(time.time() - t0 - t_read, 1)},
               "granule": {"modis": ms.granule, "viirs": gv.sw.granule}}, open(out / "stats" / f"{pair_id}.json", "w"))
    return pair_id, "ok", len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--raw")
    ap.add_argument("--out")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--download", action="store_true", help="fetch the granules of each pair first and delete them after tiling")
    a = ap.parse_args()
    raw = Path(a.raw) if a.raw else data_root() / "raw"
    out = Path(a.out) if a.out else data_root() / "episode"
    (out / "stats").mkdir(parents=True, exist_ok=True)
    pairs = read_pairs(a.pairs)
    if a.only:
        pairs = pairs[pairs.pair_id.isin(a.only)]
    rows = {r["pair_id"]: r for r in pairs.to_dict("records")}
    rows = {p: r for p, r in rows.items() if not (out / "stats" / f"{p}.json").exists()}  # finished pairs are neither downloaded nor rebuilt
    ids = sorted(rows, key=lambda p: (rows[p]["daynight"] != "night", p))  # night first
    if a.download:
        session = login()
    t0 = time.time()
    for i in range(0, len(ids), a.workers):
        sub = ids[i:i + a.workers]
        if a.download:
            for p in sub:
                download_pair(session, rows[p], raw)
        with ProcessPoolExecutor(a.workers) as ex:
            futs = {ex.submit(run_job, p, str(raw), str(out), a.device): p for p in sub}
            for f in as_completed(futs):
                try:
                    pid, st, nt = f.result()
                    print(f"[{time.time() - t0:6.0f}s] {pid} {st} {nt} tiles", flush=True)
                    if a.download:
                        shutil.rmtree(raw / pid, ignore_errors=True)
                except Exception:
                    traceback.print_exc()
                    open(out / "build_fail.log", "a").write(futs[f] + "\n")


if __name__ == "__main__":
    main()
