"""Training loop: AMP, channels_last, optional torch.compile, grad clipping, cosine schedule, best-by-fire-excess checkpoint, resume,
deterministic seeds, one-line status file per epoch.

Precision: network trunk under autocast (bf16 default; fp16 + GradScaler optional). Kept in fp32 always: input log/ratio features,
local median, anchor/background exp, Planck band conversion, footprint scatter, log radiance residuals and the loss, parameters and
optimizer state. (Radiance spans ~5 decades 4 um, 1e-1..1e2; bf16 has 8 mantissa bits = 0.4 % steps, i.e. ~0.1 K at 4 um bg but
far too coarse for 1e-3 residuals.)"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch

from nightrad.raw.core import RunConfig, forward_loss
from nightrad.raw.metrics import Acc, evaluate_batch, summarize
from nightrad.raw.net import RawHarmonizer
from nightrad.raw.schema import split_batch


def seed_all(seed: int, deterministic: bool = False):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def epoch_generator(loader):
    """The generator that drives the sampling order of a DataLoader, wherever it lives: loader.generator (shuffle), the weighted sampler's (region balance) or the batch sampler's / its sampler's (fire-tile batches)."""
    for obj in (loader, getattr(loader, "sampler", None), getattr(loader, "batch_sampler", None), getattr(getattr(loader, "batch_sampler", None), "sampler", None)):
        g = getattr(obj, "generator", None)
        if g is not None:
            return g
    return None


def make_param_groups(model, wd: float, head_lr_mult: float = 1.0):
    """the excess path (decoder output head + rho gain) may get its own learning-rate multiplier; 1.0 = one group, exactly as before."""
    m = getattr(model, "_orig_mod", model)
    if head_lr_mult == 1.0:
        return [{"params": list(model.parameters()), "weight_decay": wd, "lr_mult": 1.0}]
    head = list(m.decoder.head.parameters()) + ([m.rho_gain] if m.rho_gain is not None else [])
    ids = {id(p) for p in head}
    rest = [p for p in model.parameters() if id(p) not in ids]
    return [{"params": rest, "weight_decay": wd, "lr_mult": 1.0}, {"params": head, "weight_decay": wd, "lr_mult": head_lr_mult}]


def lr_at(step, total, warmup, base, floor=0.02, const=False):
    if step < warmup:
        return base * (step + 1) / warmup
    if const:  # DIAG: constant learning rate after warm-up
        return base
    p = (step - warmup) / max(1, total - warmup)
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


@torch.no_grad()
def validate(model, loader, cfg: RunConfig, device, amp_dtype, seed=1234):
    model.eval()
    acc = Acc()
    gen = torch.Generator(device=device).manual_seed(seed)  # same hiding every time -> comparable epochs
    tot, n = 0.0, 0
    for b in loader:
        inp, ev = split_batch(b, device)
        loss, _, aux = forward_loss(model, inp, cfg, gen, amp_dtype, return_aux=True)
        tot += float(loss) * inp["viirs_L"].shape[0]; n += inp["viirs_L"].shape[0]
        evaluate_batch(aux, inp, cfg, acc, ev)
    s = summarize(acc)
    s["val_loss"] = tot / max(n, 1)
    model.train()
    return s


def run_diag(a, model, vl, val_ds, cfg, dev, amp_dtype, epoch, step, proto=True):
    """DIAG: full-validation loss by term and by cell group, gradient shares, protocol subset; appended to diag.jsonl (new field only)."""
    from nightrad.raw import diag
    t1 = time.time()
    dg = diag.full_eval_weighted(model, vl, cfg, dev, amp_dtype)
    if getattr(a, "diag_val4", False):  # validation loss on the 4 um terms only (same hiding draw, same data)
        if cfg.net.out4:
            dg["val4"] = dg["val_total"]
        else:
            import copy
            c4 = copy.deepcopy(cfg); c4.loss.band_weights = (1.0, 0.0)
            dg["val4"] = diag.full_eval(model, vl, c4, dev, amp_dtype)["val_total"]
    if getattr(a, "diag_grad", False):
        dg["grad"] = diag.grad_shares(model, vl, cfg, dev, amp_dtype, n_batches=2)
    if proto and getattr(a, "diag_proto", True):
        if not hasattr(a, "_proto"):
            a._proto = diag.ProtoSubset(val_ds, every=getattr(a, "diag_tile_every", 1))
        dg["proto"] = a._proto(model, cfg, dev, amp_dtype)
    dg["diag_s"] = time.time() - t1
    with open(os.path.join(a.out, "diag.jsonl"), "a") as f:
        f.write(json.dumps({"epoch": epoch, "step": step, **dg}, default=float) + "\n")
    return dg


def save_ckpt(path, model, opt, sched_step, epoch, best, cfg, metrics, scaler=None):
    m = getattr(model, "_orig_mod", model)
    torch.save({"model": m.state_dict(), "opt": opt.state_dict() if opt else None, "step": sched_step, "epoch": epoch, "best": best,
                "cfg": cfg.to_dict(), "bands": __import__("nightrad.raw.planck", fromlist=["x"]).BANDS_NAME, "metrics": metrics, "scaler": scaler.state_dict() if scaler else None}, path + ".tmp")
    os.replace(path + ".tmp", path)


def train(a, train_ds, val_ds, cfg: RunConfig, log=print):
    """a: namespace with out, epochs, bs, lr, wd, warmup, amp, channels_last, compile, workers, seed, resume, deterministic, clip. Returns the best metrics dict."""
    from nightrad.raw.data import make_loader
    seed_all(a.seed, a.deterministic)
    dev = torch.device(os.environ.get("RAW_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(a.out, exist_ok=True)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}[a.amp] if dev.type == "cuda" else None
    model = RawHarmonizer(cfg.net).to(dev)
    if a.channels_last and dev.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        model.channels_last = True
    net = torch.compile(model) if a.compile and dev.type == "cuda" else model
    opt = torch.optim.AdamW(make_param_groups(model, a.wd, getattr(a, "head_lr_mult", 1.0)), lr=a.lr, weight_decay=a.wd, fused=dev.type == "cuda")
    scaler = torch.amp.GradScaler() if amp_dtype == torch.float16 else None
    sw = None
    if getattr(a, "region_balance", False) and hasattr(train_ds, "region_ids"):
        from nightrad.raw.data import region_balanced_weights
        sw = region_balanced_weights(train_ds.region_ids())
    if getattr(a, "sample_weights", None):  # big data: per-tile manifest weights
        sw = np.load(a.sample_weights).astype(np.float64)
        assert len(sw) == len(train_ds) and (sw >= 0).all() and sw.sum() > 0, "sample-weights must be one non-negative weight per train tile"
    bsamp = None
    if getattr(a, "fire_batch_frac", 0.0) > 0:  # NIGHT: fire tiles (chosen by the MODIS input only) in every batch
        from nightrad.raw.data import FireBatchSampler, modis_fire_flags, region_balanced_weights
        ff = modis_fire_flags(train_ds, cfg, dev)
        w0 = sw if sw is not None else np.ones(len(train_ds))
        bsamp = FireBatchSampler(w0 / w0.sum(), ff, a.bs, a.fire_batch_frac, a.seed)
        log(f"fire-batch sampler: {int(ff.sum())}/{len(ff)} tiles flagged (MODIS input), {bsamp.nf} per batch of {a.bs}")
    tl = make_loader(train_ds, a.bs, a.workers, True, a.seed, drop_last=True, sample_weights=sw, batch_sampler=bsamp)
    vl = make_loader(val_ds, a.bs, a.workers // 2 if a.workers < 4 else max(1, a.workers // 2), False, a.seed)
    total = a.epochs * len(tl)
    step, epoch0, best = 0, 0, {"fire_err": float("inf")}
    last = os.path.join(a.out, "last.pt")
    if a.resume and os.path.exists(last):
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        if scaler and ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        step, epoch0, best = ck["step"], ck["epoch"] + 1, ck["best"]
        log(f"resumed from epoch {ck['epoch']} step {step}")
    from nightrad.raw.data import CudaPrefetcher
    tl_iter = (lambda: CudaPrefetcher(tl, dev)) if dev.type == "cuda" else (lambda: tl)
    if getattr(a, "diag_every", 0) and epoch0 == 0:  # DIAG: the untrained network (step 0)
        run_diag(a, model, vl, val_ds, cfg, dev, amp_dtype, 0, 0, proto=True)
    for epoch in range(epoch0, a.epochs):
        t0 = time.time()
        # one hiding generator per epoch, seeded -> a resumed run draws the same masks as an uninterrupted one
        gen = torch.Generator(device=dev).manual_seed(a.seed * 100003 + epoch)
        if bsamp is not None:
            bsamp.set_epoch(epoch)
        else:
            eg = epoch_generator(tl)
            if eg is None:
                raise RuntimeError("training loader has no seedable generator (epochs would not be reproducible)")
            eg.manual_seed(a.seed * 7919 + epoch)
        model.train()
        sums: dict = {}
        ntiles = 0
        for b in tl_iter():
            inp, _ = split_batch(b, dev)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, a.warmup, a.lr, const=getattr(a, 'lr_const', False)) * g.get("lr_mult", 1.0)
            loss, logs = forward_loss(net, inp, cfg, gen, amp_dtype)
            opt.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward(); scaler.unscale_(opt)
            else:
                loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            if scaler:
                scaler.step(opt); scaler.update()
            else:
                opt.step()
            for k, v in logs.items():
                if k.endswith(("/loss", "fire_cell", "bg_cell", "nll")) or "share_" in k or k.startswith("hurdle/") or k == "total":
                    sums[k] = sums.get(k, 0) + v.detach().float()
            sums["gnorm"] = sums.get("gnorm", 0) + gn.detach()
            step += 1; ntiles += inp["viirs_L"].shape[0]
            if getattr(a, "log_every", 0) and step % a.log_every == 0:  # per-step group shares / term losses (one sync every log_every steps)
                with open(os.path.join(a.out, "steps.jsonl"), "a") as f:
                    f.write(json.dumps({"step": step, **{k: float(v) for k, v in logs.items() if k.endswith(("/loss",)) or "share_" in k or k == "total"}}) + "\n")
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_train = time.time() - t0
        nb = max(len(tl), 1)
        row = {"epoch": epoch, "step": step, "lr": lr_at(step, total, a.warmup, a.lr, const=getattr(a, 'lr_const', False)), "train_s": t_train, "tiles_s": ntiles / t_train,
               **{k: float(v) / nb for k, v in sums.items()}}
        nolegacy = getattr(a, "no_legacy_val", False)  # DIAG: skip the retired per-epoch legacy validation (final-epoch checkpoint is what is scored)
        mt = validate(model, vl, cfg, dev, amp_dtype) if not (getattr(a, "halves", None) or nolegacy) else {}  # sweep 1: ONE evaluation (sweep_eval) only
        if not (getattr(a, "halves", None) or nolegacy):  # (sweep 1 drops the legacy val/* fields: its only evaluation is row["sweep"])
            row.update({f"val/{k}": v for k, v in mt.items()})
        sw = None
        if getattr(a, "halves", None) and ((epoch + 1) % max(1, getattr(a, "eval_every", 1)) == 0 or epoch == a.epochs - 1):
            from nightrad.raw import sweep_eval
            if not hasattr(a, "_halves"):
                a._halves = json.load(open(a.halves)); a._best_key = list(best.get("key") or [9, float("inf")])  # (resume keeps the stored best key)
            sw = sweep_eval.summarize(sweep_eval.evaluate(model, vl, cfg, dev, amp_dtype), a._halves)
            row["sweep"] = sw
            key = sweep_eval.select_key(sw, epoch + 1)
            is_best = key < a._best_key
            if is_best:
                a._best_key = key
        elif getattr(a, "halves", None) or nolegacy:
            is_best = False
        else:
            sel = getattr(a, "select", "fire_err")
            is_best = mt[sel] < best.get(sel, float("inf"))
        if is_best:
            best = {"epoch": epoch} if getattr(a, "halves", None) else {"fire_err": mt["fire_err"], "epoch": epoch, **mt}
            if sw:
                best["sweep"] = sw; best["key"] = a._best_key
            save_ckpt(os.path.join(a.out, "best.pt"), model, None, step, epoch, best, cfg, mt)
        save_ckpt(last, model, opt, step, epoch, best, cfg, mt, scaler)
        if getattr(a, "ckpt_every_epoch", False):
            save_ckpt(os.path.join(a.out, f"ep{epoch + 1:03d}.pt"), model, None, step, epoch, best, cfg, {}, None)
        if not (getattr(a, "halves", None) or nolegacy):
            row["best_fire_err"] = best["fire_err"]
        row["is_best"] = is_best
        if not math.isfinite(row.get("total", 0.0)):
            raise FloatingPointError(f"non-finite training loss at epoch {epoch + 1}: stopping (NaN stop)")
        mr = sw["ALL"]["mass_ratio"] if sw else mt.get("excess_mass_ratio_meas")  # collapse detector: live for the sweep too (predicted/measured excess mass)
        collapsed = (mr is not None and mr == mr and mr < getattr(a, "stop_mass", 0.2)) and epoch >= getattr(a, "min_epochs", 3)
        row["excess_collapse"] = bool(collapsed)
        de = getattr(a, "diag_every", 0)
        if de and ((epoch + 1) % de == 0 or epoch == a.epochs - 1):
            row["diag"] = run_diag(a, model, vl, val_ds, cfg, dev, amp_dtype, epoch + 1, step, proto=((epoch + 1) // de) % max(1, getattr(a, "diag_proto_every", 1)) == 0 or epoch == a.epochs - 1)
        stop_pat = False
        if row.get("diag"):  # DIAG: keep the lowest full-validation-loss checkpoint; stop when it has not improved for --stop-patience evaluations
            vt = row["diag"]["val_total"]
            if vt < getattr(a, "_best_vt", float("inf")):
                a._best_vt, a._bad = vt, 0
                save_ckpt(os.path.join(a.out, "best_val.pt"), model, None, step, epoch, {"val_total": vt, "epoch": epoch + 1}, cfg, {}, None)
                row["is_best_val"] = True
            else:
                a._bad = getattr(a, "_bad", 0) + 1
            if getattr(a, "stop_patience", 0) and a._bad >= a.stop_patience:
                stop_pat = True
        if dev.type == "cuda":
            row["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
        with open(os.path.join(a.out, "log.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
        swa = (lambda sp: sw[sp]["mv3_fire|model"]["median"]) if sw else None
        dgl = (f"val_loss={row['diag']['val_total']:.4f} bright_share={row['diag'].get('loss_share_bright') or 0:.2f} " + (f"auc={row['diag']['proto']['auc_model']:.3f} e_msel={row['diag']['proto']['energy_msel'] or 0:.2f} top1={row['diag']['proto'].get('top1_model', 0):.2f} " if row["diag"].get("proto") else "")) if "diag" in row else ""
        line = (dgl + f"epoch={epoch + 1}/{a.epochs} step={step} loss={row.get('total', 0):.4f} "
                + (f"mv3A={swa('A'):.3f} mv3B={swa('B'):.3f} mass_ratio_A={sw['A']['mass_ratio']:.2f} admissibleA={sw['A']['admissible']['all']} best_key={getattr(a, '_best_key', None)}@{best.get('epoch', -1) + 1} " if sw else
                   "" if (getattr(a, "halves", None) or nolegacy) else
                   f"val_fire_err={mt['fire_err']:.4f} mass_ratio={mt.get('excess_mass_ratio_meas', float('nan')):.2f}{' EXCESS_COLLAPSE' if collapsed else ''} best={best['fire_err']:.4f}@{best.get('epoch', -1) + 1} ")
                + f"tiles_s={row['tiles_s']:.0f} lr={row['lr']:.2e} t={time.strftime('%H:%M:%S')}")
        open(os.path.join(a.out, "status.txt"), "w").write(line + "\n")
        log(line)
        if stop_pat:
            log(f"stopping: full-validation loss not improved for {a.stop_patience} evaluations")
            break
        if getattr(a, "stop_after", None) and epoch + 1 >= a.stop_after:  # simulate an interrupted run (tests)
            break
        if collapsed and getattr(a, "stop_on_collapse", False):
            log("stopping: excess collapsed to ~0 (excess-mass stop rule)")
            break
    if getattr(a, "halves", None) and row.get("sweep"):
        best = {**best, "final_epoch": epoch + 1, "final_sweep": row["sweep"]}
    if row.get("diag"):
        best = {**best, "final_epoch": epoch + 1, "final_diag": row["diag"]}
    json.dump(best, open(os.path.join(a.out, "results.json"), "w"), indent=1, default=float)
    return best
