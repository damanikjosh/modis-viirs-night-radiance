#!/usr/bin/env python
"""Layer step 2: Aqua night granules (catalogue.py output) -> flagged VIIRS-like 375 m cells, one parquet per granule.

usage: worker.py --catalogue catalogue.jsonl --bbox W,S,E,N [--region NAME] [--root DIR] [--ckpt weights/best_val.pt] [--device auto|cuda|cpu]
                 [--streams 3] [--batch 16] [--margin 12] [--tile-rule modis_firelevel] [--limit N] [--force] [--delete-raw]

Inputs per granule: MYD021KM + MYD03 radiance and geometry; MYD14 only for its cloud / water / quality classes (fire detections are not an input). They are downloaded
from LAADS (laads.py: NR_EARTHDATA_TOKEN or ~/.netrc) into <root>/raw/nasa/<PRODUCT>/ and reused when already there.
Outputs (root = --root, default $NR_DATA_ROOT/layer, default ./data/layer):
  pixels/year=YYYY/month=MM/<id>.parquet   flagged cells of the granule (only when >= 1 cell is flagged)
  _done/<id>.json                          status ok: stats, input sha256, parquet sha256, pipeline_version      (granule is skipped on re-run when ok at this version or newer)
  _failed/<id>.json                        status failed: error, attempt counter (removed when the granule later succeeds)
The worker refuses any checkpoint whose sha256 is not the one in nightrad.inference.version."""
import argparse
import json
import os
import socket
import sys
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import laads  # noqa: E402
from nightrad.inference.version import CKPT_EPOCH, CKPT_SHA256, DEFAULT_CKPT, PIPELINE_VERSION, sha256_file, verify_checkpoint, vkey  # noqa: E402

TILE_RULES = ("all_land", "modis_firelevel", "modis_firelevel_nbr8")


def data_root() -> Path:
    return Path(os.environ.get("NR_DATA_ROOT", "data"))


def fetch_granule(rec: dict, raw_root: Path, session):
    """Download the three input files of a granule into raw_root/<PRODUCT>/. -> (files dict key -> Path, error or None)."""
    files = {}
    for key in ("l1b", "geo", "fm"):
        r = rec.get(key)
        if not r:
            return files, f"missing {key} in catalogue"
        dst = raw_root / r["name"].split(".")[0] / r["name"]
        ok, code = laads.fetch(session, r["https"], dst, float(r.get("size_mb") or 0))
        if not ok:
            return files, f"{key} download failed: {code}"
        files[key] = dst
    return files, None


def timed_fetch(rec, raw_root, session):
    t = time.time()
    files, err = fetch_granule(rec, raw_root, session)
    return time.time() - t, files, err


def put_record(out: Path, rec: dict):
    """Write _done (ok) or _failed. An ok record of a newer pipeline version is never overwritten; a failed record never replaces an ok one of this version or newer."""
    kd, kf = out / "_done" / f"{rec['id']}.json", out / "_failed" / f"{rec['id']}.json"
    kd.parent.mkdir(parents=True, exist_ok=True); kf.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(kd.read_text()) if kd.exists() else None
    if rec["status"] == "ok":
        if old and vkey(old.get("pipeline_version")) > vkey(PIPELINE_VERSION):
            return
        kd.write_text(json.dumps(rec))
        kf.unlink(missing_ok=True)
    else:
        if old and old.get("status") == "ok" and vkey(old.get("pipeline_version")) >= vkey(PIPELINE_VERSION):
            return
        prev = json.loads(kf.read_text()) if kf.exists() else None
        rec["attempt"] = (prev.get("attempt", 1) + 1) if prev else 1
        kf.write_text(json.dumps(rec))


def is_done(out: Path, gid: str) -> bool:
    f = out / "_done" / f"{gid}.json"
    if not f.exists():
        return False
    d = json.loads(f.read_text())
    return d.get("status") == "ok" and vkey(d.get("pipeline_version")) >= vkey(PIPELINE_VERSION)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--catalogue", required=True, help="jsonl from catalogue.py")
    ap.add_argument("--bbox", required=True, help="W,S,E,N: cells outside are dropped; a catalogue line may carry its own 'bbox'")
    ap.add_argument("--region", default="region", help="free label written into every record and cell row")
    ap.add_argument("--root", default=None, help="output directory (default $NR_DATA_ROOT/layer)")
    ap.add_argument("--raw", default=None, help="raw download cache (default $NR_DATA_ROOT/raw/nasa)")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--streams", type=int, default=3, help="parallel downloads (prefetch)")
    ap.add_argument("--batch", type=int, default=16, help="tiles per model call; changes the output bit-wise, keep 16")
    ap.add_argument("--margin", type=int, default=12, help="tile margin in 375 m cells (0 = non-overlapping tiles)")
    ap.add_argument("--tile-rule", default="modis_firelevel", choices=TILE_RULES)
    ap.add_argument("--input-screen-t4", type=float, default=None, help="keep a flag only if its own MODIS pixel T4 >= this (K)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="re-process granules that are already done")
    ap.add_argument("--delete-raw", action="store_true", help="delete the downloaded inputs of a granule after it was processed")
    ap.add_argument("--name", default=socket.gethostname())
    a = ap.parse_args()
    verify_checkpoint(a.ckpt)
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from nightrad.data.modis_l1b_only import read_modis_production
    from nightrad.inference.batch import process_granule
    from nightrad.inference.infer import load_model

    root = Path(a.root) if a.root else data_root() / "layer"
    raw_root = Path(a.raw) if a.raw else data_root() / "raw" / "nasa"
    bbox = tuple(float(x) for x in a.bbox.split(","))
    recs = [json.loads(line) for line in open(a.catalogue) if line.strip()]
    if a.limit:
        recs = recs[:a.limit]
    todo = [r for r in recs if a.force or not is_done(root, r["id"])]
    dev = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else ("cpu" if a.device == "auto" else a.device))
    model, cfg, _, info = load_model(a.ckpt, dev)
    assert info["epoch"] == CKPT_EPOCH
    print(f"{len(recs)} granules in the catalogue, {len(todo)} to do, device {dev}", flush=True)
    sess = laads.session()
    n_ok = n_fail = n_flag = 0
    with ThreadPoolExecutor(max(1, a.streams)) as ex:
        it, window = iter(todo), deque()
        for _ in range(max(1, a.streams) * 2):
            r = next(it, None)
            if r is not None:
                window.append((r, ex.submit(timed_fetch, r, raw_root, sess)))
        while window:
            r, fut = window.popleft()
            nxt = next(it, None)
            if nxt is not None:
                window.append((nxt, ex.submit(timed_fetch, nxt, raw_root, sess)))
            dl_s, files, err = fut.result()
            rec = {"pipeline_version": PIPELINE_VERSION, "id": r["id"], "region": a.region, "t": r["t"], "worker": a.name, "ckpt_sha256": CKPT_SHA256,
                   "bbox": list(r.get("bbox") or bbox), "margin": a.margin, "tile_rule": a.tile_rule, "input_screen_T4_K": a.input_screen_t4}
            if err:
                rec.update(status="failed", error=err)
            else:
                try:
                    tp = time.time()
                    sw = read_modis_production(files["l1b"], files["geo"], files["fm"])
                    for attempt in range(6):  # CUDA out of memory on a shared GPU: wait and retry with the SAME batch size (outputs depend on batch composition)
                        try:
                            cols, st = process_granule(sw, tuple(rec["bbox"]), model, cfg, dev, batch=a.batch, margin=a.margin, tile_rule=a.tile_rule, input_screen_T4_K=a.input_screen_t4)
                            break
                        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                            if "out of memory" not in str(e).lower() and "ALLOC_FAILED" not in str(e) or attempt == 5:
                                raise
                            torch.cuda.empty_cache(); time.sleep(5 + 5 * attempt)
                    rec.update(status="ok", **st, inputs={k: {"name": files[k].name, "bytes": files[k].stat().st_size, "sha256": sha256_file(files[k])} for k in files})
                    if cols is not None and st["flagged"] > 0:
                        d = datetime.fromtimestamp(float(np.min(cols["utc"])), timezone.utc)
                        cols = dict(cols); n = len(cols["lon"])
                        tbl = pa.table({"lon": cols.pop("lon"), "lat": cols.pop("lat"), "utc": pa.array(cols.pop("utc"), pa.float64()), "granule": pa.array([r["id"]] * n), "region": pa.array([a.region] * n), **cols})
                        tbl = tbl.replace_schema_metadata({"pipeline_version": PIPELINE_VERSION, "checkpoint_sha256": CKPT_SHA256, "granule": r["id"], "product": "harmonized VIIRS cells EXPERIMENTAL",
                                                           "margin": str(a.margin), "tile_rule": a.tile_rule, "input_screen_T4_K": str(a.input_screen_t4)})
                        rel = f"pixels/year={d.year:04d}/month={d.month:02d}/{r['id']}.parquet"
                        out = root / rel; out.parent.mkdir(parents=True, exist_ok=True)
                        pq.write_table(tbl, out, compression="zstd")
                        rec["parquet"] = {"key": rel, "bytes": out.stat().st_size, "sha256": sha256_file(out)}
                    rec["seconds"] = {"download": round(dl_s, 1), "process": round(time.time() - tp, 1)}
                    n_flag += st["flagged"]
                except Exception as e:  # recorded, not hidden
                    rec.update(status="failed", error=f"{type(e).__name__}: {e} @ " + " | ".join(traceback.format_exc().splitlines()[-4:-1])[:420])
            put_record(root, rec)
            n_ok += rec["status"] == "ok"; n_fail += rec["status"] != "ok"
            print(f"{r['id']} {rec['status']} flagged={rec.get('flagged', '-')} {rec.get('error', '')}", flush=True)
            if a.delete_raw and files:
                for f in files.values():
                    f.unlink(missing_ok=True)
    print(f"worker finished: ok {n_ok} failed {n_fail} flagged cells {n_flag}", flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
