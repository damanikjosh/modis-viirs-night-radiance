#!/usr/bin/env python
"""Placement probe: train the U-Net on the within-pixel SHARE map only (MODIS input; target = measured VIIRS excess share) and score hottest-cell
hit rate vs chance / pixel-centre / point-spread-overlap heuristics on old validation and val2-far (>= 60 km from old train). Pair-clustered ranges.
  python scripts/placement_probe.py --train <packed> [--extra-train <packed> ...] --val <packed val> --val2 <packed val2> --out R/placement-x-s0 --seed 0 [--steps 1500]
Test split closed (prereg guard). Output: out/{last.pt, log.jsonl, score_val.json, score_val2far.json}."""
import argparse, json, os, time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from nightrad.raw import planck, prereg, placement as PL, protocol_eval as PE
from nightrad.raw.core import Prep, RunConfig
from nightrad.raw.data import CombinedShards, PackedShards, make_loader
from nightrad.raw.net import RawHarmonizer
from nightrad.raw.schema import split_batch
from nightrad.raw.sweep_eval import ref_cfg
from nightrad.raw.train import lr_at, seed_all

METHODS = ("model", "centre", "psf", "psf_nbr", "chance")


def method_scores(model, P, inp, cfg, dev, ablate=None):
    fp = P.fp
    exc = torch.exp(P.logL["modis"][:, 0]) - torch.exp(P.med["modis"][:, 0])
    exc = (exc.clamp(min=0) * (inp["modis_sat"] == 0)) * P.ok_foot
    out = {"centre": PL.centre_score(fp), "psf": PL.psf_score(fp, exc, True), "psf_nbr": PL.psf_score(fp, exc, False), "chance": torch.zeros_like(exc)}
    if model is not None:
        out["model"] = PL.predict_logits(model, P, cfg, torch.bfloat16 if dev.type == "cuda" else None, ablate)
    return out


@torch.no_grad()
def score(model, tiles, cfg, dev, bs=8, seed=0, ablate=None):
    rows = {m: [] for m in METHODS}; loss_sum, loss_n = 0.0, 0.0
    if model is not None:
        model.eval()
    n = tiles["viirs_L"].shape[0]; pairs = tiles["eval_pair_id"].reshape(n, -1)[:, 0].long()
    for i0 in range(0, n, bs):
        idx = np.arange(i0, min(n, i0 + bs)); batch = {k: v[idx] for k, v in tiles.items()}
        inp, _ = split_batch(batch, dev); P = Prep(inp, cfg); e = PL.meas_excess(P); selc = PL.select_cells(P, inp, e)
        if not selc.any():
            continue
        sc = method_scores(model, P, inp, cfg, dev, ablate)
        if "model" in sc:
            loss_sum += float(PL.share_loss(sc["model"], P.fp, e, selc)) * float((selc.float() / P.fp.total(torch.ones_like(e)[:, None])[:, 0].clamp(min=1)).sum()); loss_n += float((selc.float() / P.fp.total(torch.ones_like(e)[:, None])[:, 0].clamp(min=1)).sum())
        for m, s in sc.items():
            rows[m].append(PL.footprint_metrics(s, P.fp, e, selc, pairs[idx], seed))
    if model is not None:
        model.train()
    out = {m: {k: np.concatenate([r[k] for r in v]) for k in v[0]} for m, v in rows.items() if v}
    return out, (loss_sum / loss_n if loss_n else float("nan"))


def stats(rows, tiles, min_fp=10, n_boot=4000):
    meta = {}
    if "eval_episode" in tiles:
        n = len(tiles["eval_pair_id"]); pa = tiles["eval_pair_id"].reshape(n, -1)[:, 0].numpy(); ep = tiles["eval_episode"].reshape(n, -1)[:, 0].numpy(); rg = tiles["eval_region"].reshape(n, -1)[:, 0].numpy()
        meta = {int(p): (int(e), int(r)) for p, e, r in zip(pa, ep, rg)}
    out = {"n_footprints": int(len(rows["model"]["hit"])) if "model" in rows else int(len(rows["chance"]["hit"])), "methods": {}, "vs": {}}
    per = {}
    for m, r in rows.items():
        d = {}
        for p in np.unique(r["pair"]):
            k = r["pair"] == p
            d[int(p)] = {"n": int(k.sum()), "hit": float(r["hit"][k].mean()), "top1": float(r["top1"][k].mean()), "top3": float(r["top3"][k].mean()), "chance": float((1.0 / r["n_cells"][k]).mean()), "max_share": float(r["max_share"][k].mean())}
        per[m] = d
        q = {p: v for p, v in d.items() if v["n"] >= min_fp}
        out["methods"][m] = {"n_pairs": len(q), "pooled_hit": float(r["hit"].mean()), "pooled_top1": float(r["top1"].mean()), "pooled_top3": float(r["top3"].mean()),
                             "mean_hit": float(np.mean([v["hit"] for v in q.values()])) if q else None, "mean_top1": float(np.mean([v["top1"] for v in q.values()])) if q else None,
                             "mean_top3": float(np.mean([v["top3"] for v in q.values()])) if q else None, "chance_hit": float(np.mean([v["chance"] for v in q.values()])) if q else None,
                             "measured_hottest_share": float(np.mean([v["max_share"] for v in q.values()])) if q else None, "pairs": {str(p): v for p, v in q.items()},
                             "mean_cells": float(r["n_cells"].mean())}
        if meta and q:
            out["methods"][m]["region_balanced_hit"] = PE.region_balanced({p: v["hit"] for p, v in q.items()}, meta)["value"]
    pr_all = {m: r["pair"] for m, r in rows.items()}
    if rows:
        ups = np.unique(next(iter(rows.values()))["pair"]); rng = np.random.default_rng(0)
        agg = {m: {k: np.array([[r[k][r["pair"] == u].sum(), (r["pair"] == u).sum()] for u in ups]) for k in ("hit", "top1", "top3")} for m, r in rows.items()}
        bi = rng.integers(0, len(ups), (n_boot, len(ups))); out["pooled_clustered"] = {}
        for k in ("hit", "top1", "top3"):
            val = {m: agg[m][k][bi].sum(1) for m in agg}
            rate = {m: val[m][:, 0] / val[m][:, 1] for m in agg}
            out["pooled_clustered"][k] = {m: {"value": float(agg[m][k][:, 0].sum() / agg[m][k][:, 1].sum()), "lo": float(np.percentile(rate[m], 2.5)), "hi": float(np.percentile(rate[m], 97.5))} for m in agg}
            if "model" in agg:
                out["pooled_clustered"][k + "_diff"] = {m: {"diff": float(agg["model"][k][:, 0].sum() / agg["model"][k][:, 1].sum() - agg[m][k][:, 0].sum() / agg[m][k][:, 1].sum()), "lo": float(np.percentile(rate["model"] - rate[m], 2.5)), "hi": float(np.percentile(rate["model"] - rate[m], 97.5))} for m in agg if m != "model"}
        out["n_pairs_all"] = int(len(ups))
        ch = np.concatenate([1.0 / rows["chance"]["n_cells"]]); out["pooled_clustered"]["analytic_chance_hit"] = float(ch.mean())
    if "model" in per:
        for m in per:
            if m == "model":
                continue
            for key in ("hit", "top1", "top3"):
                a = {p: v[key] for p, v in per["model"].items() if v["n"] >= min_fp}; b = {p: v[key] for p, v in per[m].items() if v["n"] >= min_fp}
                out["vs"][f"model-{m}|{key}"] = PE.boot_diff(a, b, n_boot)
            a = {p: v["hit"] for p, v in per["model"].items() if v["n"] >= min_fp}; c = {p: v["chance"] for p, v in per["model"].items() if v["n"] >= min_fp}
        a = {p: v["hit"] for p, v in per["model"].items() if v["n"] >= min_fp}; c = {p: v["chance"] for p, v in per["model"].items() if v["n"] >= min_fp}
        out["vs"]["model-analytic_chance|hit"] = PE.boot_diff(a, c, n_boot)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True); p.add_argument("--extra-train", nargs="*", default=[]); p.add_argument("--val", required=True); p.add_argument("--val2", required=True)
    p.add_argument("--out", required=True); p.add_argument("--seed", type=int, default=0); p.add_argument("--steps", type=int, default=1500); p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--wd", type=float, default=1e-4); p.add_argument("--warmup", type=int, default=50); p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=250); p.add_argument("--workers", type=int, default=4)
    p.add_argument("--ablate", default=None, choices=[None, "geometry_only", "radiance_only", "modis_mask"], help="control: which input channels are removed (see placement.predict_logits)"); p.add_argument("--score-only", action="store_true", help="score last.pt in --out without training")
    a = p.parse_args()
    for d in [a.train, a.val, a.val2] + a.extra_train:
        prereg.open_split(d)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); os.makedirs(a.out, exist_ok=True)
    planck.use_bands("centre"); cfg = ref_cfg(RunConfig(alloc=True)); seed_all(a.seed)
    val_t = PE.load_all(PackedShards(a.val)); v2 = PE.load_all(PackedShards(a.val2))
    keep = v2["eval_far_km"].reshape(len(v2["eval_far_km"]), -1)[:, 0] >= 60.0; v2 = {k: v[keep] for k, v in v2.items()}
    print("val tiles", len(val_t["viirs_L"]), "val2-far tiles", len(v2["viirs_L"]), flush=True)
    model = RawHarmonizer(cfg.net).to(dev)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    if not a.score_only:
        ds = CombinedShards([a.train] + a.extra_train) if a.extra_train else PackedShards(a.train)
        keep_i = []; i = 0
        with torch.no_grad():
            for b in DataLoader(ds, batch_size=32, num_workers=a.workers):
                inp, _ = split_batch(b, dev); P = Prep(inp, cfg); e = PL.meas_excess(P); s = PL.select_cells(P, inp, e).flatten(1).any(1).cpu().numpy()
                keep_i += [i + j for j in np.flatnonzero(s)]; i += len(s)
        print(f"train tiles {len(ds)}, with >=1 selected footprint {len(keep_i)}", flush=True)
        if not keep_i:
            raise SystemExit("no training tile has a selected (fire-level, unsaturated, complete) MODIS footprint: nothing to train the probe on")
        sub = Subset(ds, keep_i); tl = make_loader(sub, a.bs, a.workers, True, a.seed, drop_last=True)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd, fused=dev.type == "cuda"); amp = torch.bfloat16 if dev.type == "cuda" else None
        step, t0 = 0, time.time(); model.train()
        while step < a.steps:
            for b in tl:
                inp, _ = split_batch({k: v.to(dev) for k, v in b.items()}, dev); P = Prep(inp, cfg); e = PL.meas_excess(P); selc = PL.select_cells(P, inp, e)
                if not selc.any():
                    continue
                for g in opt.param_groups:
                    g["lr"] = lr_at(step, a.steps, a.warmup, a.lr)
                logits = PL.predict_logits(model, P, cfg, amp, a.ablate)
                loss = PL.share_loss(logits, P.fp, e, selc)
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip); opt.step(); step += 1
                if step % 25 == 0:
                    log.write(json.dumps({"step": step, "train_loss": float(loss.detach()), "t": time.time() - t0}) + "\n"); log.flush()
                if step % a.eval_every == 0 or step == a.steps:
                    r, vl = score(model, val_t, cfg, dev, seed=a.seed, ablate=a.ablate); st = stats({"model": r["model"], "chance": r["chance"]}, val_t, n_boot=200)
                    row = {"step": step, "val_share_loss": vl, "val_hit": st["methods"]["model"]["mean_hit"], "val_chance": st["methods"]["model"]["chance_hit"], "val_top1": st["methods"]["model"]["mean_top1"]}
                    r2, vl2 = score(model, v2, cfg, dev, seed=a.seed, ablate=a.ablate); st2 = stats({"model": r2["model"], "chance": r2["chance"]}, v2, n_boot=200)
                    row.update({"val2far_share_loss": vl2, "val2far_hit": st2["methods"]["model"]["mean_hit"], "val2far_pooled_hit": st2["methods"]["model"]["pooled_hit"], "val2far_top1": st2["methods"]["model"]["mean_top1"], "train_loss": float(loss)})
                    log.write(json.dumps(row) + "\n"); log.flush(); print(row, flush=True)
                if step >= a.steps:
                    break
        torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(), "step": step, "seed": a.seed}, os.path.join(a.out, "last.pt"))
    else:
        model.load_state_dict(torch.load(os.path.join(a.out, "last.pt"), map_location=dev, weights_only=False)["model"])
    for name, tiles in (("val", val_t), ("val2far", v2)):
        r, vl = score(model, tiles, cfg, dev, seed=a.seed, ablate=a.ablate); res = stats(r, tiles); res["val_share_loss"] = vl; res["seed"] = a.seed; res["git"] = prereg.git_sha()
        json.dump(res, open(os.path.join(a.out, f"score_{name}.json"), "w"), default=float)
        print(name, "n_footprints", res["n_footprints"], {m: (round(v["mean_hit"], 3) if v["mean_hit"] is not None else None) for m, v in res["methods"].items()}, "n_pairs", res["methods"]["model"]["n_pairs"], flush=True)


if __name__ == "__main__":
    main()
