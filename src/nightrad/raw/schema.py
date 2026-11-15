"""Schema adapter: the ONLY file that knows how a stored tile maps onto the model's fields.

Contract (per tile, numpy; s in {modis, viirs}); the v3 tile SCHEMA note is mapped in `adapt_v3_tile`.
  {s}_L        [2,H,W] f32 radiance W m-2 sr-1 um-1: row 0 = L4, row 1 = L11. NaN allowed where code != 0.
  {s}_code     [H,W] u8 validity class: 0 valid, 1 cloud, 2 water/non-land, 3 bad/missing/bow-tie, 4 not covered, 5 glint/other
  {s}_sat      [H,W] u8 4 um channel saturated
  {s}_band21   [H,W] u8 MODIS: band 21 used for L4 (0 for VIIRS)
  {s}_psize    [2,H,W] f32 native pixel size along/across scan, km
  {s}_vza      [H,W] f32 view zenith, degrees
  {s}_cossza   [H,W] f32 cos solar zenith
  modis_pid    [H,W] i32 source native pixel id of each cell, DENSE 0..n-1, -1 = none (VIIRS: cell is its own pixel)
  modis_nbr    [4,H,W] i32 OPTIONAL dense ids of the pixels next to each cell's pixel (along-scan -/+, along-track -/+; -1 none):
               only for the weighted MODIS view and the 3x3 scorer; build with neighbour_table() from native ids line*1354+sample
  modis_complete [H,W] u8 footprint not cut by the tile border (constant per footprint)
  eval_*       anything: evaluation only, NEVER an input or a target (the model cannot receive them)
"""
from __future__ import annotations

import numpy as np
import torch

SENSORS = ("modis", "viirs")
EVAL_PREFIX = "eval_"
INPUT_KEYS = (
    [f"{s}_{k}" for s in SENSORS for k in ("L", "code", "sat", "band21", "psize", "vza", "cossza")]
    + ["modis_pid", "modis_complete"]
)

# ---- v3 (schema_version 5) key map, from docs/tile_schema.md; verified on the real 50-tile sample ----
V3_KEYS = {"L4": "{s}_rad4", "L11": "{s}_rad11", "vclass": "{s}_vclass", "sat": "{s}_sat", "t4_band": "{s}_t4_band", "psize": "{s}_psize",
           "vza": "{s}_vza", "cossza": "{s}_cossza", "pid": "modis_pid", "complete": "modis_footprint_complete"}
# v3 vclass (0 off-swath, 1 not covered, 2 clear land incl. fire, 3 cloud, 4 water, 5 bad/missing, 6 sun glint) -> model code
# (0 valid, 1 cloud, 2 water, 3 bad, 4 not covered, 5 glint)
_VCLASS_TO_CODE = np.array([4, 4, 0, 1, 2, 3, 5], np.uint8)


def densify_pid(pid: np.ndarray) -> np.ndarray:
    """Arbitrary native-pixel ids (-1 = none) -> dense 0..n-1 per tile (-1 kept)."""
    out = np.full(pid.shape, -1, np.int32)
    ok = pid >= 0
    if ok.any():
        out[ok] = np.unique(pid[ok], return_inverse=True)[1].astype(np.int32)
    return out


def neighbour_table(pid_raw: np.ndarray, dense: np.ndarray, stride: int = 1354) -> np.ndarray:
    """Native ids (line*stride + sample, -1 none) -> [4,H,W] dense neighbour ids: sample-1, sample+1 (along scan), line-1, line+1 (along track)."""
    out = np.full((4, *pid_raw.shape), -1, np.int32)
    ok = pid_raw >= 0
    if not ok.any():
        return out
    ids, first = np.unique(pid_raw[ok], return_index=True)
    dd = dense[ok][first]
    for k, off in enumerate((-1, 1, -stride, stride)):
        q = pid_raw + off
        pos = np.clip(np.searchsorted(ids, q), 0, len(ids) - 1)
        hit = ok & (ids[pos] == q)
        out[k][hit] = dd[pos][hit]
    return out


# VIIRS I4 at the placeholder ceiling. In the v3 tiles the stored VIIRS rad4 is Planck(367 K, 3.74 um) on (a) every VIIRS cell the FIRE PRODUCT declares saturated (all of
# them product fire pixels: 326 of 326 on a 10 % sample of train) and (b) every vclass-5 cell ("not processed", 33,006 on the sample). That value is written by the tile builder
# from product information, NOT measured, so it is PRODUCT-DERIVED. Policy (F1/F2, review): such cells are removed by the mask (code 5), never fed as signal,
# and the VIIRS `sat` channel is not used (set to 0). The test is on the stored VALUE, not on the product flag.
VIIRS_I4_CEILING = 4.562311667043153  # Planck(367 K, 3.74 um), W m-2 sr-1 um-1


def adapt_v3_tile(z) -> dict:
    """Stored v3 tile (npz-like mapping) -> sample dict in the contract above. Copies every `eval_*` key through unchanged (they are split off by
    split_batch and never reach the model). Missing radiance (0 in the tile) is left 0: such cells have code != 0 and are masked everywhere."""
    g = lambda k, s="modis": np.asarray(z[V3_KEYS[k].format(s=s)])
    out = {}
    for s in SENSORS:
        out[f"{s}_L"] = np.stack([g("L4", s), g("L11", s)]).astype(np.float32)
        out[f"{s}_code"] = _VCLASS_TO_CODE[np.clip(g("vclass", s), 0, 6)]
        out[f"{s}_sat"] = g("sat", s).astype(np.uint8)
        out[f"{s}_band21"] = (g("t4_band", s) == 21).astype(np.uint8)  # MODIS band 21 behind L4; VIIRS t4_band is 0
        out[f"{s}_psize"] = g("psize", s).astype(np.float32)
        out[f"{s}_vza"] = g("vza", s).astype(np.float32)
        out[f"{s}_cossza"] = g("cossza", s).astype(np.float32)
        if s == "viirs":
            at_ceiling = np.isclose(out["viirs_L"][0], VIIRS_I4_CEILING, rtol=1e-4)
            out["viirs_code"] = np.where(at_ceiling, np.uint8(5), out["viirs_code"]).astype(np.uint8)
            out["viirs_sat"] = np.zeros_like(out["viirs_sat"])  # the stored flag IS the fire product's declaration: never an input
        bad = out[f"{s}_code"] != 0  # whatever the stored values are outside the mask, the sample carries zeros there
        out[f"{s}_L"] = np.where(bad[None], np.float32(0), out[f"{s}_L"])
        out[f"{s}_sat"] = np.where(bad, np.uint8(0), out[f"{s}_sat"])
    raw = g("pid")
    out["modis_pid"] = densify_pid(raw)
    out["modis_nbr"] = neighbour_table(raw, out["modis_pid"])
    out["modis_complete"] = g("complete").astype(np.uint8)
    for k in z.keys():
        if k.startswith(EVAL_PREFIX):
            out[k] = np.asarray(z[k])
    # evaluation-only side information (numeric; the model never receives eval_*): pair id for clustered bootstraps, time gap, scan angles, day flag
    out["eval_pair_id"] = np.int32((100000 if str(z["pair_id"])[0] == "N" else 0) + int(str(z["pair_id"])[1:])) if "pair_id" in z else np.int32(-1)  # N-pairs (night-fire set) live above 100000 so they never collide with the P-pairs
    out["eval_dt_min"] = np.float32(z["dt_min"]) if "dt_min" in z else np.float32(np.nan)
    out["eval_daynight"] = np.uint8(z["daynight"]) if "daynight" in z else np.uint8(255)
    for s in SENSORS:
        if f"{s}_scan" in z:
            out[f"eval_{s}_scan"] = np.asarray(z[f"{s}_scan"], np.float32)
    return out


def split_batch(batch: dict, device=None):
    """-> (inputs dict with exactly INPUT_KEYS, eval dict). Model-side code only ever sees the first."""
    keys = INPUT_KEYS + (["modis_nbr"] if "modis_nbr" in batch else [])
    inp = {k: batch[k].to(device, non_blocking=True) if device else batch[k] for k in keys}
    ev = {k: v.to(device, non_blocking=True) if device else v for k, v in batch.items() if k.startswith(EVAL_PREFIX)}
    return inp, ev
