#!/usr/bin/env python
"""Train the raw-signal model (one run).
  python scripts/train_raw.py --data synthetic --out runs_local/syn --epochs 20            # synthetic tiles with known truth (smoke test, CPU)
  python scripts/train_raw.py --data $NR_DATA_ROOT/train_night --val-data $NR_DATA_ROOT/val2_night --out runs/<name> ...   # packed real tiles (docs/model.md)
Data directories are packed shard directories (nightrad.raw.data.pack_dataset). Paths are arguments; NR_DATA_ROOT is only a shell convenience."""
import argparse
import hashlib
import json
import os
import subprocess
import sys

from nightrad.raw.core import RunConfig
from nightrad.raw.data import PackedShards
from nightrad.raw.hide import HideConfig
from nightrad.raw.nf2_split import file_sha16
from nightrad.raw.losses import LossConfig, shares_for_fire
from nightrad.raw.net import NetConfig
from nightrad.raw.synth import SyntheticTiles
from nightrad.raw.train import train


def make_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="synthetic"); p.add_argument("--val-data", default=None)
    p.add_argument("--n-train", type=int, default=2048); p.add_argument("--n-val", type=int, default=256)
    p.add_argument("--tile", type=int, default=96)
    p.add_argument("--out", required=True); p.add_argument("--epochs", type=int, default=20); p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3); p.add_argument("--wd", type=float, default=1e-4); p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"]); p.add_argument("--no-channels-last", dest="channels_last", action="store_false")
    p.add_argument("--compile", action="store_true"); p.add_argument("--compile-prep", action="store_true", help="also compile the masked median"); p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--resume", action="store_true"); p.add_argument("--deterministic", action="store_true")
    p.add_argument("--width", type=int, default=32); p.add_argument("--latent", type=int, default=16)
    p.add_argument("--huber-delta", type=float, default=0.10); p.add_argument("--no-hide", action="store_true")
    p.add_argument("--hide-window", type=int, default=9); p.add_argument("--hide-top", type=int, default=3); p.add_argument("--hide-p", type=float, default=0.5)
    p.add_argument("--t-fire", type=float, default=800.0); p.add_argument("--widen", type=int, default=0)
    p.add_argument("--min-valid", type=float, default=0.10); p.add_argument("--alloc", action="store_true"); p.add_argument("--alloc-weight", type=float, default=0.1)
    # switchable options from the adversarial reviews; defaults = the plan
    p.add_argument("--mv-mode", default="free", choices=["free", "hurdle", "measured"], help="M->V total: free | hurdle (predicted) | measured (ablation)")
    p.add_argument("--kernel", default="tri", choices=["box", "tri"]); p.add_argument("--w-scan", type=float, default=0.001); p.add_argument("--w-track", type=float, default=0.0)
    p.add_argument("--no-class-balance", action="store_true", help="old fire weighting instead of fixed group shares"); p.add_argument("--fire-share", type=float, default=0.4, choices=[0.2, 0.4, 0.6], help="pre-declared grid; rest split bg:near = 2:1")
    p.add_argument("--w-mv", type=float, default=1.0); p.add_argument("--w-vm", type=float, default=0.5); p.add_argument("--w-mm", type=float, default=0.1); p.add_argument("--w-vv", type=float, default=0.1)
    p.add_argument("--log-every", type=int, default=25, help="steps between per-step log lines (group shares etc.) in steps.jsonl")
    p.add_argument("--delta-lowres", type=int, default=1); p.add_argument("--delta-max", type=float, default=0.25); p.add_argument("--tie11", action="store_true")
    p.add_argument("--arch", default="unet", choices=["unet", "unet_nobottleneck_skip", "conv3"]); p.add_argument("--sun-term", action="store_true")
    p.add_argument("--daynight", default="night", choices=["both", "night", "day"]); p.add_argument("--partial-p", type=float, default=0.0)
    p.add_argument("--stop-on-collapse", action="store_true"); p.add_argument("--stop-mass", type=float, default=0.2); p.add_argument("--min-epochs", type=int, default=3)
    p.add_argument("--allow-test", action="store_true"); p.add_argument("--frozen", default="configs/raw.json")
    p.add_argument("--select", default="fire_err", help="validation measure that picks best.pt (lower is better), e.g. mv3_fire")
    p.add_argument("--class-shares", type=float, nargs=3, default=None, help="bg near fire shares (overrides --fire-share)")
    p.add_argument("--fire-level", type=float, default=0.5); p.add_argument("--no-rho-skip", action="store_true")
    p.add_argument("--block-w", type=float, default=0.0, help="3x3-block loss weight, M->V and V->M (sweep 1; 0 = off)")
    p.add_argument("--halves", default=None, help="sweep 1: JSON {A:[pairs],B:[pairs]}; evaluates with sweep_eval each epoch, selects best.pt on half A")
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--w-energy", type=float, default=0.0, help="energy-consistency term on the M->V path (sweep 1; 0 = off)"); p.add_argument("--false-share", type=float, default=0.0, help="4th loss group: false excess under bright MODIS pixels (share; 0 = off)")
    p.add_argument("--hide-jitter", type=int, default=0, help="random shift of the hiding window centre (cells); 0 = old behaviour")
    p.add_argument("--loss", default="legacy", choices=["legacy", "gamma"], help="legacy = group-share Huber on log (reference); gamma = mean-consistent Gamma deviance on linear radiance")
    p.add_argument("--loss-family", default="gamma", choices=["gamma", "poisson", "nmse"], help="with --loss gamma: deviance family")
    p.add_argument("--input-weight", type=float, default=0.0, help="gamma mode: extra weight on M->V cells under bright MODIS pixels (input-side only)")
    p.add_argument("--scale-w", type=float, nargs=3, default=[1.0, 1.0, 1.0], help="gamma: weights of cell / MODIS-pixel / 3x3-block deviance on the M->V path")
    p.add_argument("--extra-data", nargs="*", default=[], help="more packed TRAIN night sets to combine with --data (e.g. the night-fire train pack); only common keys are used")
    p.add_argument("--region-balance", action="store_true", help="draw training tiles so every region (a pseudo-region per set without labels) contributes equally; needs --extra-data")
    p.add_argument("--fire-batch-frac", type=float, default=0.0, help="share of every batch drawn from tiles with a MODIS fire-level pixel (flag from the MODIS input only); 0 = off")
    p.add_argument("--bands", default="centre", choices=["centre", "data"], help="sweep 1 factor: effective wavelengths of the Planck conversions (centre = frozen default)")
    p.add_argument("--diag-every", type=int, default=0, help="every N epochs (and at step 0 and the last epoch) log the full-validation loss by term / by MODIS-bright vs other cells to diag.jsonl; 0 = off")
    p.add_argument("--diag-proto-every", type=int, default=1, help="protocol subset (block AUC, hottest-cell share, energy) on every k-th diag evaluation (and the last)")
    p.add_argument("--diag-tile-every", type=int, default=1, help="use every k-th validation tile for the protocol subset")
    p.add_argument("--diag-grad", action="store_true", help="gradient shares of bright vs other cells at each diag evaluation")
    p.add_argument("--no-legacy-val", action="store_true", help="skip the retired per-epoch legacy validation (no best.pt; last.pt is the result)")
    p.add_argument("--stop-patience", type=int, default=0, help="stop when the full-validation loss has not improved for this many diag evaluations (0 = off); best_val.pt = lowest")
    p.add_argument("--val-select", default=None, help="split JSON with episode lists (configs/nf2_split_episodes.json or configs/val2_split.json): validate only on the tiles of the --val-side episodes of --val-data (needs episodes.json and eval_episode.npy in the pack)")
    p.add_argument("--val-side", default="select", choices=["select", "report"])
    p.add_argument("--git-commit", default=None, help="exact code commit (default: env GIT_COMMIT or git rev-parse); written to run_meta.json before step 1")
    p.add_argument("--sample-weights", default=None, help="big data: .npy of one sampling weight per TRAIN tile (pack order), e.g. sample_weight x region_weight of the nightfire2 manifest; replaces --region-balance")
    p.add_argument("--val-list", default=None, help="big data: configs/nf2_split.json; validate only on the tiles of --val-side listed there (pack needs names.json); REPORT is refused")
    p.add_argument("--ckpt-every-epoch", action="store_true", help="save model-only ep<N>.pt after every epoch so the validation curve can be re-scored offline")
    p.add_argument("--bright-weight", type=float, default=0.0, help="M->V weight 1 + G*clamp(rho4_modis,0,3) on cell/pixel/block terms; MODIS input only")
    p.add_argument("--head-lr-mult", type=float, default=1.0, help="learning-rate multiplier of the excess path (decoder output head and rho gain)")
    p.add_argument("--out-bands", default="both", choices=["both", "4"], help="'4' = the decoder, scene, views and every reconstruction term cover ONLY the 4 um band (inputs stay both bands); 'both' (default) = unchanged")
    p.add_argument("--diag-val4", action="store_true", help="also log val4 = the validation loss on the 4 um terms only (for a two-band model: the same loss with the 11 um band weight 0; for --out-bands 4 it equals val_total), the one loss comparable across the two kinds of model")
    p.add_argument("--lr-const", action="store_true", help="constant learning rate after warm-up (no cosine)")
    return p


def make_cfg(a):
    from nightrad.raw import planck
    planck.use_bands(a.bands)
    from nightrad.raw.views import ViewConfig
    return RunConfig(net=NetConfig(width=a.width, latent=a.latent, out4=a.out_bands == "4", arch=a.arch, delta_lowres=a.delta_lowres, delta_max=a.delta_max, sun_term=a.sun_term, rho_skip=not a.no_rho_skip),
                    loss=LossConfig(delta=a.huber_delta, alloc_weight=a.alloc_weight, class_balance=not a.no_class_balance, class_shares=tuple(a.class_shares) if a.class_shares else shares_for_fire(a.fire_share), fire_level=a.fire_level, block_w_mv=a.block_w, block_w_vm=a.block_w, w_energy=a.w_energy, mode=a.loss, loss_family=a.loss_family, input_weight=a.input_weight, bright_weight=a.bright_weight, scale_w=tuple(a.scale_w), false_share=a.false_share, term_weights={"mv": a.w_mv, "vm": a.w_vm, "mm": a.w_mm, "vv": a.w_vv}),
                    hide=HideConfig(enabled=not a.no_hide, window=a.hide_window, n_top=a.hide_top, p_tile=a.hide_p, partial_p=a.partial_p, center_jitter=a.hide_jitter),
                    view=ViewConfig(t_fire=a.t_fire, kernel=a.kernel, w_scan=a.w_scan, w_track=a.w_track), widen=a.widen, min_valid_frac=a.min_valid,
                    alloc=a.alloc, mv_mode=a.mv_mode, tie11=a.tie11, daynight=a.daynight)


def expand_preset(argv):
    """--preset NAME: prepend the flags of configs/paper_runs.json[NAME]; flags given on the command line come later and override them."""
    if "--preset" not in argv:
        return argv
    i = argv.index("--preset")
    name = argv[i + 1]
    presets = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "paper_runs.json")))["presets"]
    if name not in presets:
        raise SystemExit(f"unknown preset {name!r}; known: {sorted(presets)}")
    return presets[name]["args"] + argv[:i] + argv[i + 2:]


def main(argv=None):
    a = make_parser().parse_args(expand_preset(list(sys.argv[1:] if argv is None else argv)))
    cfg = make_cfg(a)
    from nightrad.raw import prereg
    for d in [a.data, a.val_data] + list(a.extra_data):  # tuning uses train + val only: a test split needs --allow-test + frozen config, and is logged
        if d and d != "synthetic":
            prereg.open_split(d, allow_test=a.allow_test, frozen_path=a.frozen, cfg=cfg.to_dict(), ckpt=None)
    if a.data == "synthetic":
        tr, va = SyntheticTiles(a.n_train, 0, a.tile, a.tile, cache=True), SyntheticTiles(a.n_val, 10**6, a.tile, a.tile, cache=True)
    else:
        from nightrad.raw.data import CombinedShards
        va_idx = None
        if a.val_select:
            import numpy as np
            sel = set(json.load(open(a.val_select))[a.val_side]); epn = json.load(open(os.path.join(a.val_data, "episodes.json")))
            keep = {v for k, v in epn.items() if k in sel}
            ev = np.load(os.path.join(a.val_data, "eval_episode.npy")); ev = ev.reshape(len(ev), -1)[:, 0]
            va_idx = [i for i, e in enumerate(ev) if int(e) in keep]
            print("VAL-SELECT", a.val_side, len(va_idx), "tiles of", len(ev), flush=True)
        if a.val_list:
            from nightrad.raw.nf2_split import side_indices
            assert a.val_side == "select", "validation during training uses SELECT only; REPORT is for stage 5"
            va_idx = side_indices(json.load(open(os.path.join(a.val_data, "names.json"))), a.val_list, a.val_side)
            print("VAL-LIST", a.val_side, len(va_idx), "tiles", flush=True)
        tr, va = (CombinedShards([a.data] + a.extra_data) if a.extra_data else PackedShards(a.data)), PackedShards(a.val_data or a.data, va_idx)
        for d in (a.data, a.val_data):  # held-out episode exclusion applied when packing (scripts/pack_v3.py): log it at start-up
            r = os.path.join(d or "", "pack_report.json")
            print("EXCLUSION", d, json.dumps({k: v for k, v in json.load(open(r)).items() if k != "episodes"}) if os.path.exists(r) else "NO pack_report.json (not packed with pack_v3.py)", flush=True)
    if a.compile_prep:
        from nightrad.raw import background
        background.enable_compile()
    # run folder metadata BEFORE step 1: exact commit, config, data-list hashes, seed
    commit = a.git_commit or os.environ.get("GIT_COMMIT")
    if not commit:
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            commit = None
    if not commit:
        raise SystemExit("no git commit: pass --git-commit or set GIT_COMMIT (never 'unknown')")
    def _h(path):
        f = os.path.join(path, "meta.json")
        return hashlib.sha256(open(f, "rb").read()).hexdigest()[:16] if path and os.path.exists(f) else None
    os.makedirs(a.out, exist_ok=True)
    json.dump({"git_commit": commit, "seed": a.seed, "cfg": cfg.to_dict(), "args": vars(a),
               "data": {d: {"meta_sha256_16": _h(d), "pack_report": json.load(open(os.path.join(d, "pack_report.json"))) if os.path.exists(os.path.join(d, "pack_report.json")) else None} for d in [a.data, a.val_data] + list(a.extra_data) if d and d != "synthetic"},
               "lists": {"val_list_sha256_16": file_sha16(a.val_list) if a.val_list else None, "sample_weights_sha256_16": file_sha16(a.sample_weights) if a.sample_weights else None,
                         "train_names_sha256_16": file_sha16(os.path.join(a.data, "names.json")) if a.data != "synthetic" and os.path.exists(os.path.join(a.data, "names.json")) else None,
                         "val_names_sha256_16": file_sha16(os.path.join(a.val_data, "names.json")) if a.val_data and os.path.exists(os.path.join(a.val_data, "names.json")) else None, "val_tiles_used": len(va)},
               "val_select_sha256_16": file_sha16(a.val_select) if a.val_select else None,
               "hide_jitter_cells": a.hide_jitter, "deterministic": a.deterministic, "started": __import__("time").strftime("%Y-%m-%d %H:%M:%S")}, open(os.path.join(a.out, "run_meta.json"), "w"), indent=1, default=str)
    print(json.dumps({"cfg": cfg.to_dict(), "args": vars(a)}))
    train(a, tr, va, cfg)


if __name__ == "__main__":
    main()
