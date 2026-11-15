"""Finalize the built tiles: exclusions, fire episodes, train / val2 split by 100 km block, weights, manifest, leak check.

  python scripts/finalize.py --pairs PAIRS.csv [--build DIR] [--out DIR] [--workers 8]

Reads stats/*.json and tiles/<pair>/*.npz of scripts/build_tiles.py and writes
  <out>/{train,val2}/<tile>.npz     tiles with final `split` ('train' | 'val'), `holdout` ('' | 'val2') and `sample_weight`
  <out>/manifest.parquet            kept tiles (episode, group_name, split, sample_weight, region_weight, ...)
  <out>/manifest_all.parquet        every built tile with `keep` and `drop_reason`
  <out>/episodes.parquet, build_summary.json, leak_check.json
Steps (rules in nightrad.data.splits): 1. drop tiles in an exclusion zone or near an exclusion centre; 2. tile-level episodes;
3. spatial blocks of 100 km are given whole to train or val2 (about 30 % of each region's fire weight to val2, seed search);
4. train tiles within 60 km of a val2 tile are dropped; 5. stratum weights per split (sample_weight) and region weights;
6. independent recomputation of every leak rule on the final manifest (`leak_check.json`, `ok` must be true).
Defaults: build = $NR_DATA_ROOT/tiles_build, out = $NR_DATA_ROOT/tiles."""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from nightrad.data import splits as S
from nightrad.data.download import data_root
from nightrad.data.tables import read_centres, read_pairs


def rewrite(job):
    src, dst, w, sp = job
    with np.load(src, allow_pickle=False) as d:
        t = {k: d[k] for k in d.files}
    t["sample_weight"] = np.float32(w)
    t["split"], t["holdout"] = np.array("val" if sp == "val2" else "train"), np.array("val2" if sp == "val2" else "")
    np.savez_compressed(dst, **t)
    return os.path.getsize(dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--build")
    ap.add_argument("--out")
    ap.add_argument("--centres", help="exclusion centres csv (lat, lon); default: the packaged list")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    build = Path(a.build) if a.build else data_root() / "tiles_build"
    out = Path(a.out) if a.out else data_root() / "tiles"
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:5.0f}s] {s}", flush=True)
    pm = read_pairs(a.pairs)[["pair_id", "region"]].rename(columns={"region": "nf_region"})
    rows = []
    for p in sorted((build / "stats").glob("*.json")):
        rows += json.load(open(p))["rows"]
    df = pd.DataFrame(rows).merge(pm, on="pair_id", how="inner").reset_index(drop=True)
    df["region_cont"] = df.region  # continent code of the tile builder
    log(f"{len(df)} tiles from {df.pair_id.nunique()} pairs")
    centres = read_centres(a.centres)
    why, dist = S.exclusion_reason(df, centres)
    df["dist_exclusion_centre_km"], df["drop_reason"], df["keep"] = dist, why, why == ""
    k = df[df.keep].copy()
    k["episode"] = ["E%04d" % e for e in S.episodes(k)]
    k["group_name"] = S.group_of(k.nf_region)
    iv, sinfo = S.choose_split(k)
    log(f"split search: {sinfo}")
    k["split"], k["holdout"] = np.where(iv, "val2", "train"), np.where(iv, "val2", "")
    v2, tr = k[k.split == "val2"], (k.split == "train").to_numpy()
    d2 = np.full(len(k), np.inf)
    if len(v2) and tr.any():
        d2[tr] = S.min_dist_km(k.lat.values[tr], k.lon.values[tr], v2.lat.values, v2.lon.values)
    k["dist_val2_km"] = d2
    buf_drop = tr & (d2 < S.BUFFER_KM)
    df.loc[k.index[buf_drop], ["keep", "drop_reason"]] = [False, "within_60km_of_val2_tile"]
    df = df.merge(k[["tile", "episode", "split"]], on="tile", how="left")
    k = k[~buf_drop].copy()
    log(f"episodes {k.episode.nunique()}; train tiles dropped by the val2 buffer: {int(buf_drop.sum())}")
    for sp in ("train", "val2"):
        m = (k.split == sp).to_numpy()
        k.loc[m, "sample_weight"] = S.stratum_weights(np.full(m.sum(), sp), k.has_fire[m].values.astype(bool), k.random_tile[m].values.astype(bool))
    k["region_weight"] = S.region_weights(k)
    k["path"] = [f"{sp}/{t}.npz" for sp, t in zip(k.split, k.tile)]
    for sp in ("train", "val2"):
        (out / sp).mkdir(parents=True, exist_ok=True)
    jobs = [(str(build / "tiles" / r.pair_id / f"{r.tile}.npz"), str(out / r.path), float(r.sample_weight), r.split) for r in k.itertuples()]
    with ProcessPoolExecutor(a.workers) as ex:
        k["bytes"] = list(ex.map(rewrite, jobs, chunksize=64))
    log(f"final tiles written ({k.bytes.sum() / 1e9:.2f} GB)")
    k = k.sort_values(["split", "pair_id", "tile"]).reset_index(drop=True)
    # ---- independent leak check on the final manifest (episodes and blocks recomputed from scratch)
    T, V = k[k.split == "train"], k[k.split == "val2"]
    kd = S.min_dist_km(k.lat.values, k.lon.values, centres.lat.values, centres.lon.values)
    d_tv = S.min_dist_km(T.lat.values, T.lon.values, V.lat.values, V.lon.values) if len(T) and len(V) else np.array([np.inf])
    ep2, blk2 = S.episodes(k), S.blocks(k.lat.values, k.lon.values)
    cross = lambda lab: int(sum(1 for _, g in k.assign(l=lab).groupby("l") if g.split.nunique() > 1))
    leak = {"tiles_final": len(k), "tiles_built": len(df), "tiles_train": len(T), "tiles_val2": len(V),
            "min_dist_to_exclusion_centres_km": float(kd.min()), "min_dist_train_to_val2_km": float(d_tv.min()), "buffer_km": S.BUFFER_KM,
            "n_within_buffer_of_exclusion_centres": int((kd < S.BUFFER_KM).sum()), "n_train_within_buffer_of_val2": int((d_tv < S.BUFFER_KM).sum()),
            "n_episodes_crossing_train_val2": cross(ep2), "n_blocks_crossing_train_val2": cross(blk2),
            "pairs_with_tiles_in_both_splits": int((k.groupby("pair_id").split.nunique() > 1).sum()),
            "n_year2023": int((k.year >= 2023).sum()), "n_south_america": int((k.region_cont == "SA").sum()),
            "n_sumatra_block_plus60km": int((S.box_dist_km(k.lat, k.lon, S.INDO_BLOCK) < S.BUFFER_KM).sum()),
            "n_kalimantan_sepoct2015_150km": int((((k.year == 2015) & k.month.isin([9, 10])) & (S.box_dist_km(k.lat, k.lon, S.KALIMANTAN_BOX) < S.KALIMANTAN_BUFFER_KM)).sum()),
            "train_tiles_dropped_by_val2_buffer": int(buf_drop.sum()), "split_search": sinfo, "dropped_by_reason": df[~df.keep].drop_reason.value_counts().to_dict()}
    leak["ok"] = (leak["n_within_buffer_of_exclusion_centres"] == 0 and leak["n_train_within_buffer_of_val2"] == 0 and leak["n_episodes_crossing_train_val2"] == 0
                  and leak["n_blocks_crossing_train_val2"] == 0 and all(leak[x] == 0 for x in ("n_year2023", "n_south_america", "n_sumatra_block_plus60km", "n_kalimantan_sepoct2015_150km")))
    summ = {"pairs": int(k.pair_id.nunique()), "tiles_built": len(df), "tiles_kept": len(k), "episodes": int(k.episode.nunique()), "per_split": {}, "by_group_split": {}}
    fire = lambda g: g.modis_fire_px + g.viirs_fire_cells / 5
    for sp, g in k.groupby("split"):
        summ["per_split"][sp] = {"pairs": int(g.pair_id.nunique()), "episodes": int(g.episode.nunique()), "tiles": len(g), "night_tiles": int((g.daynight == 1).sum()),
                                 "modis_fire_px": int(g.modis_fire_px.sum()), "viirs_fire_cells": int(g.viirs_fire_cells.sum())}
    for (gn, sp), g in k.groupby(["group_name", "split"]):
        sub = k[k.split == sp]
        summ["by_group_split"][f"{gn}|{sp}"] = {"pairs": int(g.pair_id.nunique()), "episodes": int(g.episode.nunique()), "tiles": len(g), "modis_fire_px": int(g.modis_fire_px.sum()),
                                                "viirs_fire_cells": int(g.viirs_fire_cells.sum()), "share_of_split_fire_sample": float(fire(g).sum() / max(fire(sub).sum(), 1e-9)),
                                                "weighted_share": float((fire(g) * g.region_weight).sum() / max((fire(sub) * sub.region_weight).sum(), 1e-9))}
    summ["split_search"] = sinfo
    epi = k.groupby(["episode", "split"]).agg(pairs=("pair_id", "nunique"), tiles=("tile", "size"), region=("nf_region", lambda s: s.value_counts().index[0]),
                                              year=("year", "min"), month=("month", "min")).reset_index()
    json.dump(summ, open(out / "build_summary.json", "w"), indent=1)
    json.dump(leak, open(out / "leak_check.json", "w"), indent=1)
    k.to_parquet(out / "manifest.parquet", index=False)
    df.to_parquet(out / "manifest_all.parquet", index=False)
    epi.to_parquet(out / "episodes.parquet", index=False)
    log(f"done: {len(k)} kept of {len(df)}; leak ok={leak['ok']}; {json.dumps(summ['per_split'])[:300]}")
    raise SystemExit(0 if leak["ok"] else 1)


if __name__ == "__main__":
    main()
