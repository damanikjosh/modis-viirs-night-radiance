"""split the night-fire wave-2 VAL2 tiles into SELECT / REPORT by 100 km spatial block (a block is never cut; tile-level episodes live inside one block).
Rule (data properties only; no model output): blocks = connected components of tiles linked when closer than `link_km` on ANY date. Per reporting region (manifest group_name) the blocks
are visited in descending order of their fire-level MODIS pixels (ties: a seeded shuffle) and each goes to the side that currently holds fewer of that region's fire pixels (then fewer
tiles); regions are visited by descending fire, and a region's first (largest) block goes to the globally lagging side so the global fire total stays balanced. Fixed seed. Regions with one block cannot be balanced (stated in the table)."""
from __future__ import annotations

import os

import numpy as np
from scipy.spatial import cKDTree

R_KM = 6371.0088


def xyz(lat, lon):
    la, lo = np.deg2rad(np.asarray(lat, float)), np.deg2rad(np.asarray(lon, float))
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], 1)


def chord(km):
    return 2 * np.sin(km / R_KM / 2)


def blocks(lat, lon, link_km=100.0) -> np.ndarray:
    """Block id per tile: connected components of the graph 'centres closer than link_km'. Labels 0..n-1 in order of first appearance."""
    p = xyz(lat, lon); n = len(p)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, j in cKDTree(p).query_pairs(chord(link_km), output_type="ndarray"):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    roots = np.array([find(i) for i in range(n)])
    _, first = np.unique(roots, return_index=True)
    lab = {roots[i]: k for k, i in enumerate(sorted(first))}
    return np.array([lab[r] for r in roots])


def min_dist_km(lat_a, lon_a, lat_b, lon_b) -> float:
    d, _ = cKDTree(xyz(lat_b, lon_b)).query(xyz(lat_a, lon_a), k=1)
    return float(2 * R_KM * np.arcsin(np.minimum(d.min() / 2, 1.0)))


def assign(region, block, fire, tiles, seed=0):
    """Return side ('select'|'report') per BLOCK id -> dict. region/block/fire/tiles are per-tile arrays; fire = fire-level MODIS pixels of the tile."""
    rng = np.random.default_rng(seed)
    region, block, fire = map(np.asarray, (region, block, fire))
    bl = {}
    for r, b, f in zip(region, block, fire):
        d = bl.setdefault(int(b), {"region": r, "fire": 0.0, "tiles": 0}); d["fire"] += float(f); d["tiles"] += 1
    side, tot = {}, {"select": 0.0, "report": 0.0}
    rf = {r: sum(d['fire'] for d in bl.values() if d['region'] == r) for r in set(region)}
    for r in sorted(set(region), key=lambda r: (-rf[r], str(r))):  # big regions first: their coarse blocks are placed while the global total is still open
        ids = [b for b, d in bl.items() if d["region"] == r]
        ids = [ids[i] for i in rng.permutation(len(ids))]
        ids.sort(key=lambda b: -bl[b]["fire"])  # stable: ties keep the seeded shuffle
        acc = {"select": [0.0, 0], "report": [0.0, 0]}
        first = ("select", "report") if tot["select"] <= tot["report"] else ("report", "select")
        for k, b in enumerate(ids):
            if k == 0:
                s = first[0]
            else:
                s = min(("select", "report"), key=lambda x: (acc[x][0], acc[x][1], x != first[1]))
            side[b] = s; acc[s][0] += bl[b]["fire"]; acc[s][1] += bl[b]["tiles"]
        for s in acc:
            tot[s] += acc[s][0]
    return side


def side_indices(pack_names, split_json, side):
    """Indices (into a pack's names.json) of the tiles of `side` ('train'|'select'|'report') in configs/nf2_split.json. Every listed tile must be in the pack and every pack tile must be in
    exactly one list of the split (so SELECT validation can never touch REPORT or TRAIN tiles)."""
    import json
    tiles = json.load(open(split_json))["tiles"]
    owner = {t: k for k in tiles for t in tiles[k]}
    assert len(owner) == sum(len(v) for v in tiles.values()), "a tile appears in two lists"
    missing = [t for t in pack_names if t not in owner]
    assert not missing, f"{len(missing)} pack tiles are in no list (e.g. {missing[:3]})"
    return [i for i, t in enumerate(pack_names) if owner[t] == side]


def file_sha16(path):
    import hashlib
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


DEFAULT_RETIRED = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "configs", "report_retired.json")


def retired_ids(retired_json):
    import json
    return set(json.load(open(retired_json))["retired_tile_ids"])


def report_effective(split_json, retired_json):
    """REPORT as defined after the cross-check list was found to contain report tiles: the tiles of configs/nf2_split.json 'report' MINUS the retired ids (configs/report_retired.json). nf2_split.json itself is unchanged, so its sha256 pin stays valid."""
    import json
    ret = retired_ids(retired_json)
    return [t for t in json.load(open(split_json))["tiles"]["report"] if t not in ret]


def forbidden_tiles(split_json, allow_report=False, retired_json=None):
    """Tile ids a scoring run must never touch: the RETIRED tiles under ANY set (even the final REPORT scoring), and the REPORT list (closed until the final scoring) unless allow_report."""
    import json
    rep = set() if allow_report else set(json.load(open(split_json))["tiles"]["report"])
    return rep | retired_ids(retired_json or DEFAULT_RETIRED)


COMMITTED_RETIRED_SHA256 = "bf95c2f5567d2ec23cfa39e490892b8a056112b1eb7d97a071edd8aed7a262f7"   # configs/report_retired.json (88 tiles). The file used for the paper (sha256 e41d5155...2a1) differs only by wording of the free-text fields.
COMMITTED_SPLIT_SHA256 = "f978b22322cbad7eefd923637473266fff03a179718ce2d4e9d41a0cb9a8e091"   # configs/nf2_split.json. The file used for the paper (sha256 6851bb72...04c, recorded in its score files) differs only by two free-text strings; the tile and block lists are identical.


def file_sha256(path):
    import hashlib
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def check_scoring_set(pack_dir, split_json, set_name, allow_unnamed_reason=None, final=False, indices=None, split_override_reason=None, retired_json=None):
    """Scorer guard: refuse to score a pack whose tile ids (names.json) intersect the REPORT list unless this is the final stage-5 REPORT scoring
    of the REPORT set itself (set_name == 'report' and final). Only the tiles that will really be scored count (`indices`; None = the whole pack). Hardening (guard review): the split file must be the committed
    configs/nf2_split.json (sha256 pinned) unless a logged override reason is given; names.json must have exactly one id per pack tile (meta.json n); nothing to score is an error.
    A pack without names.json cannot be checked by id and is refused unless --allow-unnamed-reason documents why (old v3 validation pack: ids not in nf2, leak-checked by distance).
    Returns the record stored in every score JSON."""
    import json, os
    sha = file_sha256(split_json); retired_json = retired_json or DEFAULT_RETIRED; rsha = file_sha256(retired_json)
    rec = {"split_sha256": sha, "split_pinned": sha == COMMITTED_SPLIT_SHA256, "split_override_reason": split_override_reason, "retired_sha256": rsha, "retired_pinned": rsha == COMMITTED_RETIRED_SHA256}
    if rsha != COMMITTED_RETIRED_SHA256 and not split_override_reason:
        raise SystemExit(f"GUARD: retire list {retired_json} (sha256 {rsha[:12]}...) is not the committed configs/report_retired.json ({COMMITTED_RETIRED_SHA256[:12]}...): pass --split-override-reason to score against another file (stored in the score JSON)")
    if sha != COMMITTED_SPLIT_SHA256 and not split_override_reason:
        raise SystemExit(f"GUARD: split file {split_json} (sha256 {sha[:12]}...) is not the committed configs/nf2_split.json ({COMMITTED_SPLIT_SHA256[:12]}...): pass --split-override-reason '<why>' to score against another file (the override is stored in the score JSON)")
    if indices is not None and len(indices) == 0:
        raise SystemExit("GUARD: no tiles selected for scoring")
    f = os.path.join(pack_dir, "names.json")
    if not os.path.exists(f):
        if not allow_unnamed_reason:
            raise SystemExit(f"GUARD: {pack_dir} has no names.json: tile ids cannot be checked against the REPORT list; pass --allow-unnamed-reason '<why>' only for a pack known to be independent")
        return {**rec, "checked": False, "reason": allow_unnamed_reason, "report_hits": None, "retired_hits": None, "tiles_checked": 0}
    names = json.load(open(f)); meta = os.path.join(pack_dir, "meta.json")
    if not os.path.exists(meta):
        raise SystemExit(f"GUARD: {pack_dir} has names.json but no meta.json: cannot verify the ids file length")
    n = json.load(open(meta))["n"]
    if len(names) != n:
        raise SystemExit(f"GUARD: names.json has {len(names)} ids but the pack holds {n} tiles")
    names = set(names if indices is None else [names[i] for i in indices])   # only the tiles that will really be scored
    ret = retired_ids(retired_json); hit_ret = names & ret
    if hit_ret:
        raise SystemExit(f"GUARD: {len(hit_ret)} RETIRED tiles in the scoring set of {pack_dir} (e.g. {sorted(hit_ret)[:3]}): the 88 listed tiles are retired from REPORT and from every scoring set (decision 1, 08:38)")
    bad = names & forbidden_tiles(split_json, allow_report=(set_name == "report" and final), retired_json=retired_json)
    if bad:
        raise SystemExit(f"GUARD: {len(bad)} tiles of {pack_dir} are in the REPORT list (e.g. {sorted(bad)[:3]}): REPORT is closed until the final scoring")
    return {**rec, "checked": True, "tiles_checked": len(names), "report_hits": 0, "retired_hits": 0}
