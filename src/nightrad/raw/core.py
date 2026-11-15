"""One training/eval step: masks -> hiding -> 4 encoder passes (batched) -> scenes -> fixed views -> four loss terms.

Radiometry, medians, views and losses always run in fp32; only the network trunk may run under autocast (bf16/fp16)."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from nightrad.raw import planck
from nightrad.raw.background import masked_local_median
from nightrad.raw.hide import HideConfig, hide_regions
from nightrad.raw.losses import LossConfig, alloc_loss, apply_allocation, gamma_term, hurdle_loss, term_loss
from nightrad.raw.mask import Footprints, common_mask, dense_pid, valid_fraction_ok
from nightrad.raw.net import NetConfig, RawHarmonizer, build_features
from nightrad.raw.views import ModisView, ViewConfig, modis_to_viirs_background, tie_kappa, viirs_view

SENSORS = ("modis", "viirs")


@dataclass
class RunConfig:
    net: NetConfig = field(default_factory=NetConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    hide: HideConfig = field(default_factory=HideConfig)
    view: ViewConfig = field(default_factory=ViewConfig)  # default weighted (tri) MODIS view
    widen: int = 0  # cloud-edge widening d (cells)
    min_valid_frac: float = 0.10  # tiles with fewer common-mask cells are dropped from loss and input
    median_k: int = 7
    median_dil: int = 2
    alloc: bool = False  # Stage-3 allocation head
    mv_mode: str = "free"  # M->V total: 'free' (per-cell excess from the net) | 'hurdle' (P(fire) x amount, predicted) | 'measured' (copy the measured MODIS total: ablation only)
    tie11: bool = False  # tie the 11 um excess to the 4 um excess by Planck (always on for mv_mode != 'free')
    daynight: str = "night"  # DEFAULT night first; 'day' and 'both' are explicit switches; tiles of the other kind are dropped from input and loss

    def __post_init__(self):
        assert self.mv_mode in ("free", "hurdle", "measured") and self.daynight in ("both", "night", "day")
        assert not self.net.out4 or (self.mv_mode == "free" and not self.alloc and not self.tie11), "out4 (4 um only) supports only mv_mode 'free' without allocation or tied 11 um"
        self.net.alloc_head = self.alloc or self.mv_mode != "free"
        self.net.hurdle_head = self.mv_mode == "hurdle"
        self.net.tie11 = self.tie11 or self.mv_mode != "free"
        if self.mv_mode != "free":
            pass  # the head describes the 4 um total; 11 um excess follows physically

    def to_dict(self):
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        return cls(net=NetConfig(**d.pop("net")), loss=LossConfig(**d.pop("loss")), hide=HideConfig(**d.pop("hide")),
                   view=ViewConfig(**d.pop("view")), **d)


def day_flag(inp: dict) -> torch.Tensor:
    """[B] bool: tile is daytime (mean cos solar zenith over its cells > 0)."""
    return inp["viirs_cossza"].float().flatten(1).mean(1) > 0


class Prep:
    """Masks and measured-signal derived quantities, built from the model-input dict only (never eval_*)."""

    def __init__(self, inp: dict, cfg: RunConfig):
        self.inp = inp
        self.common = common_mask(inp["modis_code"], inp["viirs_code"], cfg.widen)
        self.common &= valid_fraction_ok(self.common, cfg.min_valid_frac)[:, None, None]
        self.day = day_flag(inp)  # [B] bool
        if cfg.daynight != "both":
            self.common &= (self.day if cfg.daynight == "day" else ~self.day)[:, None, None]
        nbr = dense_pid(inp["modis_nbr"]) if "modis_nbr" in inp else None
        self.fp = Footprints(dense_pid(inp["modis_pid"]), nbr)
        self.ok_foot = self.fp.ok(self.common, inp["modis_complete"] > 0) & self.common
        self.logL, self.med, self.rho4 = {}, {}, {}
        for s in SENSORS:
            L = inp[f"{s}_L"].float()
            m = self.common[:, None]
            self.logL[s] = torch.where(m, torch.log(torch.where(m, L, torch.ones_like(L)).clamp(min=1e-6)), torch.zeros_like(L))
            self.med[s] = masked_local_median(self.logL[s], m, cfg.median_k, cfg.median_dil)
            self.rho4[s] = torch.where(self.common, self.logL[s][:, 0] - self.med[s][:, 0], torch.zeros_like(self.common, dtype=torch.float32))


def _anchor(sensor: str, med_log: torch.Tensor, nb: int = 2) -> torch.Tensor:
    """ln background anchor in the VIIRS band from the local median of ln L in the source sensor's band. nb=1 (4 um only): the 4 um anchor alone; the 11 um median is still a
    network INPUT feature (build_features) but no 11 um background is predicted."""
    if sensor == "viirs":
        return med_log if nb == 2 else med_log[:, :nb]
    return torch.log(modis_to_viirs_background(torch.exp(med_log), nb).clamp(min=1e-12))


def _features(P: Prep, s: str, in_mask, med_log):
    inp = P.inp
    return build_features(s, inp[f"{s}_L"].float(), in_mask, med_log, inp[f"{s}_sat"], inp[f"{s}_band21"], inp[f"{s}_vza"],
                          inp[f"{s}_psize"], inp[f"{s}_cossza"])


def _noop(_name):
    pass


def run_scenes(model: RawHarmonizer, P: Prep, cfg: RunConfig, gen=None, amp_dtype=None, hidden=None, mark=_noop):
    """-> dict (sensor, 'hid'|'clean') -> dict(B, E, alloc?) linear fp32 radiance in the VIIRS band, plus the hidden masks."""
    dev = P.common.device
    if hidden is None:
        hidden = {s: (hide_regions(P.rho4[s], P.common, cfg.hide, gen, P.fp if s == "modis" else None)
                      if cfg.hide.enabled else torch.zeros_like(P.common)) for s in SENSORS}
    mark("hide")
    keys, xs, gs, ms, an, sn = [], [], [], [], [], []
    for s in SENSORS:
        variants = [("hid", P.common & ~hidden[s])] if cfg.hide.enabled else []
        variants.append(("clean", P.common))
        for name, im in variants:
            med = masked_local_median(P.logL[s], im[:, None], cfg.median_k, cfg.median_dil) if name == "hid" else P.med[s]
            x, g = _features(P, s, im, med)
            keys.append((s, name)); xs.append(x); gs.append(g); ms.append(P.common); an.append(_anchor(s, med, cfg.net.n_out)); sn.append(torch.where(P.common, P.inp[f"{s}_cossza"].float(), torch.zeros_like(P.common, dtype=torch.float32)))
    mark("features+median")
    n_modis = sum(1 for k in keys if k[0] == "modis") * P.common.shape[0]
    with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        out = model(torch.cat(xs), torch.cat(ms), torch.cat(gs), torch.cat(an), n_modis, torch.cat(sn) if cfg.net.sun_term else None)
    mark("network")
    b = P.common.shape[0]
    scenes = {}
    kap = tie_kappa(cfg.net.t_tie)
    for i, k in enumerate(keys):
        sl = slice(i * b, (i + 1) * b)
        B = torch.exp(out["logB"][sl])
        E = B * torch.expm1(out["s"][sl])
        al = out["alloc"][sl] if out["alloc"] is not None else None
        extra = {}
        if k[0] == "modis" and k[1] == "clean" and cfg.mv_mode != "free":
            logit = al
            p = torch.exp(logit - P.fp.logsumexp(logit))  # softmax over the footprint's cells (sums to 1)
            cnt = P.fp.total(torch.ones_like(logit)[:, None])[:, 0]
            if cfg.mv_mode == "hurdle":
                h = P.fp.mean(out["hurdle"][sl][:, None])[:, 0]
                a = P.fp.mean(out["amount"][sl][:, None])[:, 0]
                ebar = torch.sigmoid(h) * torch.exp(a.clamp(-12, 8))  # expected mean excess per cell (VIIRS 4 um)
                extra = {"hurdle": out["hurdle"][sl], "amount": out["amount"][sl]}
            else:  # 'measured': copy the measured MODIS total (ablation)
                Em = baseline_modis_excess(P, cfg)
                ebar = Em[:, 0]
            E4 = ebar * cnt * p
            E = torch.stack([E4, kap * E4], 1)
        else:
            if cfg.net.tie11:
                E = torch.stack([E[:, 0], kap * E[:, 0]], 1)
            if cfg.alloc and k[0] == "modis" and al is not None:
                E = apply_allocation(E, al, P.fp)
        scenes[k] = {"B": B, "E": E, "alloc": al, **extra}
    return scenes, hidden


def _tgt(x: torch.Tensor, nb: int) -> torch.Tensor:
    """Target / reference tensor [B,2,H,W] -> the predicted bands only: the whole tensor for both bands (unchanged), the 4 um slice with out4 (the 11 um values are dropped here, once)."""
    return x if nb == 2 else x[:, :nb]


def block_loss(P: "Prep", preds: dict, inp: dict, lc, cfg: "RunConfig", logs: dict):
    """3x3-native-pixel block loss on SUMS of 4 um excess (the scale at which M->V is claimed), blocks that contain a fire-level unit of the target sensor.
    Huber of ln((sum predicted + eps) / (sum measured + eps)); each footprint counts once. Predicted excess = softplus(L_pred - background) (differentiable)."""
    eps, fl = 1e-3, lc.fire_level
    fp, ok = P.fp, P.ok_foot
    w = ok.float() / fp.total(torch.ones_like(ok, dtype=torch.float32)[:, None])[:, 0].clamp(min=1.0)
    sp = lambda d: F.softplus(d * 50.0) / 50.0
    blk = lambda e: fp.sum3x3(fp.total(e[:, None])[:, 0])
    out = 0.0
    Bv, Bm = torch.exp(P.med["viirs"][:, 0]), torch.exp(P.med["modis"][:, 0])
    tv, tm = torch.exp(P.logL["viirs"][:, 0]), torch.exp(P.logL["modis"][:, 0])
    if lc.block_w_mv > 0:
        me = (tv - Bv).clamp(min=0) * ok
        pe = sp(torch.exp(preds["mv"][:, 0]) - Bv) * ok
        sel = ok & fp.any((P.rho4["viirs"] >= fl) & ok)
        r = torch.log((blk(pe) + eps) / (blk(me) + eps))
        from nightrad.raw.losses import huber
        ww = w * sel.float()
        l = (huber(r, lc.delta) * ww).sum() / ww.sum().clamp(min=1e-9)
        out = out + lc.block_w_mv * l; logs["block_mv/loss"] = l.detach()
    if lc.block_w_vm > 0:
        me = (tm - Bm).clamp(min=0) * ok
        pe = sp(torch.exp(preds["vm"][:, 0]) - Bm) * ok
        sel = ok & (P.rho4["modis"] >= fl) & (inp["modis_sat"] == 0)
        r = torch.log((fp.sum3x3(pe) + eps) / (fp.sum3x3(me) + eps))
        from nightrad.raw.losses import huber
        ww = w * sel.float()
        l = (huber(r, lc.delta) * ww).sum() / ww.sum().clamp(min=1e-9)
        out = out + lc.block_w_vm * l; logs["block_vm/loss"] = l.detach()
    return out


def gamma_total(P, inp, scenes, hidden, cfg: RunConfig, count, sat, hw, mview):
    """Gamma-deviance objective, LINEAR radiance, both bands, no target-dependent weights:
    M->V (weight term_weights['mv']): fine-cell + MODIS-pixel mean + complete 3x3-block mean deviance of the VIIRS-view prediction decoded from MODIS vs measured VIIRS, scale weights lc.scale_w;
    V->M (term_weights['vm']): deviance per footprint of the MODIS view of the VIIRS-decoded scene vs measured MODIS;
    V->V hidden (term_weights['vv']) per cell with the hiding weights; M->M hidden (term_weights['mm']) per footprint;
    energy consistency lc.w_energy: clean MODIS-decoded scene through the MODIS view vs measured MODIS per footprint (the MODIS energy principle)."""
    lc = cfg.loss; hid = "hid" if cfg.hide.enabled else "clean"; fp = P.fp
    ok = P.ok_foot; okf = ok.float()
    nb = cfg.net.n_out  # predicted bands: 2, or 1 (4 um only, ): the 11 um targets are not read
    Lv, Lm = torch.exp(_tgt(P.logL["viirs"], nb)), torch.exp(_tgt(P.logL["modis"], nb))  # measured, linear [B,nb,H,W]
    dev_ = Lv.device
    ref_v = torch.stack([planck.planck_radiance(300.0, planck.LAMBDA["viirs_4"]), planck.planck_radiance(300.0, planck.LAMBDA["viirs_11"])][:nb]).to(dev_)
    ref_m = torch.stack([planck.planck_radiance(300.0, planck.LAMBDA["modis_4"]), planck.planck_radiance(300.0, planck.LAMBDA["modis_11"])][:nb]).to(dev_)
    bg_v, bg_m = torch.exp(_tgt(P.med["viirs"], nb)), torch.exp(_tgt(P.med["modis"], nb))  # nmse: local masked-median background of the target sensor's own field (detached inside family_dev)
    wfoot = okf / count  # every footprint counts once
    logs, total = {}, 0.0
    def v_view(sc):
        return viirs_view(sc["B"], sc["E"])
    def m_view(sc):
        return mview(sc["B"], sc["E"], fp)
    # M->V at three scales
    mu = v_view(scenes[("modis", "clean")])
    w_cell = torch.ones_like(okf)
    if lc.input_weight > 0:  # input-side emphasis: cells inside MODIS fire-level footprints (measured MODIS rho4 >= fire_level, unsaturated)
        w_cell = w_cell + lc.input_weight * (ok & (P.rho4["modis"] >= lc.fire_level) & ~sat["modis"]).float()
    w_in = torch.ones_like(okf)  # smooth MODIS-input brightness weight (rho4 of the MODIS input, never a VIIRS or product quantity)
    if lc.bright_weight > 0:
        w_in = 1.0 + lc.bright_weight * P.rho4["modis"].clamp(0.0, 3.0)
        w_cell = w_cell * w_in
    d_cell = gamma_term(mu, Lv, P.common, w_cell, sat["viirs"], lc, ref_v, bg_v)
    mu_p = fp.mean(mu * P.common[:, None]); y_p = fp.mean(Lv * P.common[:, None])  # pixel mean over the cells (footprints are complete inside the mask on ok cells)
    bg_p = fp.mean(bg_v * P.common[:, None])
    d_pix = gamma_term(mu_p, y_p, ok, wfoot * w_in, torch.zeros_like(ok), lc, ref_v, bg_p)
    d_blk = torch.zeros((), device=okf.device)
    if fp.nbr is not None:
        nblk = fp.sum3x3(okf)
        cblk = ok & (nblk > 8.5)
        mu_b = torch.stack([fp.sum3x3(mu_p[:, c]) for c in range(nb)], 1) / 9.0; y_b = torch.stack([fp.sum3x3(y_p[:, c]) for c in range(nb)], 1) / 9.0
        bg_b = torch.stack([fp.sum3x3(bg_p[:, c]) for c in range(nb)], 1) / 9.0
        d_blk = gamma_term(mu_b, y_b, cblk, wfoot * w_in, torch.zeros_like(ok), lc, ref_v, bg_b)
    sw = lc.scale_w
    l_mv = (sw[0] * d_cell + sw[1] * d_pix + sw[2] * d_blk) / max(sum(sw), 1e-9)
    total = total + lc.term_weights["mv"] * l_mv
    logs.update({"mv/loss": l_mv.detach(), "mv/cell": d_cell.detach(), "mv/pix": d_pix.detach(), "mv/blk": d_blk.detach()})
    # V->M
    l_vm = gamma_term(m_view(scenes[("viirs", "clean")]), Lm, ok, wfoot, sat["modis"], lc, ref_m, bg_m)
    total = total + lc.term_weights["vm"] * l_vm; logs["vm/loss"] = l_vm.detach()
    # V->V hidden (cells; hiding weights depend on the input only)
    l_vv = gamma_term(v_view(scenes[("viirs", hid)]), Lv, P.common, hw(hidden["viirs"]), sat["viirs"], lc, ref_v, bg_v)
    total = total + lc.term_weights["vv"] * l_vv; logs["vv/loss"] = l_vv.detach()
    # M->M hidden (per footprint)
    l_mm = gamma_term(m_view(scenes[("modis", hid)]), Lm, ok, wfoot * hw(hidden["modis"]), sat["modis"], lc, ref_m, bg_m)
    total = total + lc.term_weights["mm"] * l_mm; logs["mm/loss"] = l_mm.detach()
    if lc.w_energy > 0:
        l_e = gamma_term(m_view(scenes[("modis", "clean")]), Lm, ok, wfoot, sat["modis"], lc, ref_m, bg_m)
        total = total + lc.w_energy * l_e; logs["energy/loss"] = l_e.detach()
    return total, logs


def baseline_modis_excess(P: "Prep", cfg: RunConfig) -> torch.Tensor:
    """Measured MODIS excess (L - local median)^+ converted to the VIIRS band, per cell [B,2,H,W] (constant per footprint)."""
    mv = ModisView(cfg.view)
    Lm, Bm = torch.exp(P.logL["modis"]), torch.exp(P.med["modis"])
    f = torch.tensor([mv.f4, mv.f11], device=Lm.device)[None, :, None, None]
    return (Lm - Bm).clamp(min=0) / f


def forward_loss(model, inp: dict, cfg: RunConfig, gen=None, amp_dtype=None, return_aux=False, mark=_noop):
    mark("start")
    P = Prep(inp, cfg)
    mark("masks+medians")
    scenes, hidden = run_scenes(model, P, cfg, gen, amp_dtype, mark=mark)
    return loss_from_scenes(P, scenes, hidden, inp, cfg, return_aux, mark)


def loss_from_scenes(P: "Prep", scenes: dict, hidden: dict, inp: dict, cfg: RunConfig, return_aux=False, mark=_noop):
    """The loss terms from the decoded scenes and the measured targets (split out of forward_loss so that tests can change a target without changing the scenes). With cfg.net.out4 only
    the 4 um band of every target is read: the 11 um target values never enter any term."""
    lc = cfg.loss
    nb = cfg.net.n_out
    mview = ModisView(cfg.view)
    hid = "hid" if cfg.hide.enabled else "clean"
    count = P.fp.total(torch.ones_like(P.common, dtype=torch.float32)[:, None])[:, 0].clamp(min=1.0)
    terms, preds = {}, {}
    sat = {s: inp[f"{s}_sat"] > 0 for s in SENSORS}

    def pred_v(sc):
        return torch.log(viirs_view(sc["B"], sc["E"]).clamp(min=1e-12))

    def pred_m(sc):
        return torch.log(mview(sc["B"], sc["E"], P.fp).clamp(min=1e-12))

    def hw(h):
        return torch.where(h, lc.hidden_weight, lc.visible_weight) if cfg.hide.enabled else torch.ones_like(P.common, dtype=torch.float32)

    spec = {
        "mm": ("modis", scenes[("modis", hid)], pred_m, True),
        "vv": ("viirs", scenes[("viirs", hid)], pred_v, True),
        "vm": ("modis", scenes[("viirs", "clean")], pred_m, False),
        "mv": ("viirs", scenes[("modis", "clean")], pred_v, False),
    }
    total = 0.0
    logs = {}
    if lc.mode == "gamma":
        total, logs = gamma_total(P, inp, scenes, hidden, cfg, count, sat, hw, mview)
        if return_aux:
            preds = {"mv": pred_v(scenes[("modis", "clean")]), "vm": pred_m(scenes[("viirs", "clean")]), "vv": pred_v(scenes[("viirs", hid)]), "mm": pred_m(scenes[("modis", hid)])}
        mark("views+loss")
        logs["total"] = total.detach()
        if return_aux:
            return total, logs, {"P": P, "scenes": scenes, "hidden": hidden, "preds": preds}
        return total, logs
    for name, (tgt, sc, view, same) in spec.items():
        mask = P.ok_foot if tgt == "modis" else P.common
        w = torch.ones_like(P.rho4[tgt])
        if same:
            w = w * hw(hidden[tgt])
        if tgt == "modis" and lc.footprint_norm == "unit":
            w = w / count
        pl = view(sc)
        preds[name] = pl
        fmask = None
        if name == "mv" and lc.false_share > 0:  # cells under a bright MODIS pixel (measured MODIS fire-level footprint)
            fmask = P.ok_foot & (P.rho4["modis"] >= lc.fire_level) & ~sat["modis"]
        loss, st = term_loss(pl, _tgt(P.logL[tgt], nb), mask, w, sat[tgt], lc, P.rho4[tgt], fmask)
        terms[name] = loss
        total = total + lc.term_weights[name] * loss
        logs.update({f"{name}/{k}": v for k, v in st.items()})
    if lc.w_energy > 0:  # energy consistency on the M->V path: clean MODIS-decoded scene through the MODIS view vs the measured MODIS radiance
        pl = pred_m(scenes[("modis", "clean")])
        le, st = term_loss(pl, _tgt(P.logL["modis"], nb), P.ok_foot, torch.ones_like(P.rho4["modis"]) / (count if lc.footprint_norm == "unit" else 1.0), sat["modis"], lc, P.rho4["modis"])
        total = total + lc.w_energy * le
        logs["energy/loss"] = le.detach(); logs["energy/share_fire"] = st["share_fire"]
    if (lc.block_w_mv > 0 or lc.block_w_vm > 0) and P.fp.nbr is not None:
        total = total + block_loss(P, preds, inp, lc, cfg, logs)
    if cfg.mv_mode == "hurdle":
        sc = scenes[("modis", "clean")]
        e_meas = (torch.exp(P.logL["viirs"][:, 0]) - torch.exp(P.med["viirs"][:, 0])).clamp(min=0.0)
        hl, hs = hurdle_loss(sc["hurdle"], sc["amount"], e_meas, P.fp, P.ok_foot, P.common & (P.rho4["viirs"] >= lc.fire_level), lc)
        total = total + lc.hurdle_weight * hl
        logs.update({k: v for k, v in hs.items()}); logs["hurdle/loss"] = hl.detach()
    if cfg.alloc or cfg.mv_mode != "free":
        sc = scenes[("modis", "clean")]
        e_meas = (torch.exp(P.logL["viirs"][:, 0]) - torch.exp(P.med["viirs"][:, 0])).clamp(min=0.0)
        nll, uni = alloc_loss(sc["alloc"], e_meas, P.fp, P.ok_foot)
        total = total + lc.alloc_weight * nll
        logs["alloc/nll"], logs["alloc/uniform_nll"] = nll.detach(), uni
    mark("views+loss")
    logs["total"] = total.detach() if torch.is_tensor(total) else torch.tensor(float(total))
    if return_aux:
        return total, logs, {"P": P, "scenes": scenes, "hidden": hidden, "preds": preds}
    return total, logs
