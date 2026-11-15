"""RawHarmonizer: per-sensor input layers -> SHARED masked U-Net encoder -> latent -> SHARED decoder -> per-cell scene.

Output per fine cell (VIIRS band): background = local masked median anchor x exp(bounded learned correction) and a
non-negative excess parameterised as log-ratio s >= 0:  L = B * exp(s),  E = B * expm1(s).
NOT inputs: time gap, lat/lon, date, land cover, any fire-product field. forward() takes tensors only (no **kwargs); eval_* arrays
are stripped by schema.split_batch and cannot reach the model.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nightrad.raw import planck
from nightrad.raw.layers import FiLM, MBlock, MConv, MaskCtx, mpool, mup

N_FEAT = 12  # see build_features
N_COND = 7


@dataclass
class NetConfig:
    width: int = 32
    latent: int = 16
    delta_max: float = 0.25  # bounded background correction, log units (+-28 %)
    s_max: float = 8.0  # max log-ratio excess (e^8 = 3000x)
    s_bias: float = -4.0  # initial raw excess ~ softplus(-4) = 0.018 (excess starts ~0, not at a fixed amount)
    alloc_head: bool = False  # Stage-3 softmax allocation logit
    hurdle_head: bool = False  # P(fire) logit + log amount of the footprint total (M->V head predicts the total)
    arch: str = "unet"  # 'unet' | 'unet_nobottleneck_skip' (no full-res skip: real bottleneck) | 'conv3' (3 masked convs baseline)
    delta_lowres: int = 1  # background correction predicted at 1/k resolution (k divides H, W), then replicated
    tie11: bool = False  # 11 um excess = kappa * 4 um excess, kappa = B11(t_tie)/B4(t_tie) (applied in core.run_scenes; s11 unused)
    t_tie: float = 800.0
    rho_skip: bool = True  # initialisation aid: s += gain * max(rho_in, 0), gain learnable (init 1): 'explain what you see' from step 0; hidden cells have rho=0
    sun_term: bool = False  # optional reflected-sunlight term in the day background: ln B += k_band * max(cos SZA, 0), k learnable (bounded)
    out4: bool = False  # predict ONLY the 4 um band (inputs unchanged, both bands). The decoder head has 1 background + 1 excess channel instead of 2 + 2; there is no 11 um scene at all. Default off = both bands (unchanged)

    @property
    def n_out(self) -> int:
        """Number of predicted bands (2 = 4 and 11 um, 1 = 4 um only)."""
        return 1 if self.out4 else 2


_REF = {  # ln Planck radiance at 300 K per sensor band: fixed physical constants, the compressed scale at the network door
    s: torch.stack([torch.log(planck.planck_radiance(300.0, planck.LAMBDA[f"{s}_4"])),
                    torch.log(planck.planck_radiance(300.0, planck.LAMBDA[f"{s}_11"]))]) for s in ("modis", "viirs")
}


def build_features(sensor: str, L, in_mask, med_log, sat, band21, vza, psize, cossza):
    """All inputs derived from raw radiance + conditions. L [B,2,H,W], in_mask [B,H,W] bool, med_log [B,2,H,W] (masked local median
    of ln L, hidden cells excluded). Everything outside in_mask is forced to 0 BEFORE any arithmetic (NaN-safe).
    Returns (x [B,N_FEAT,H,W] fp32, g [B,N_COND] fp32)."""
    m = in_mask[:, None]
    mf = m.float()
    lnL = torch.where(m, torch.log(torch.where(m, L, torch.ones_like(L)).clamp(min=1e-6)), torch.zeros_like(L))
    ref = _REF[sensor].to(L.device)[None, :, None, None]
    xr = torch.where(m, (lnL - ref) / 2.0, torch.zeros_like(lnL))
    rho = torch.where(m, (lnL - med_log).clamp(-3, 6) / 2.0, torch.zeros_like(lnL))
    z = lambda t, s=1.0: torch.where(in_mask, t.float() / s, torch.zeros_like(t.float()))[:, None]
    ps = torch.where(m, psize.float() / 3.0, torch.zeros_like(psize.float()))
    x = torch.cat([xr, rho, mf, z(sat), z(band21), z(vza, 60.0), ps, z(cossza)], 1)  # 2+2+1+1+1+1+2+1 = 11
    x = torch.cat([x, (xr[:, 0:1] - xr[:, 1:2])], 1)  # 4-11 um contrast
    n = mf.sum((1, 2, 3)).clamp(min=1.0)
    mean = lambda t: (t * mf).sum((1, 2, 3)) / n  # t [B,1,H,W]
    sid = torch.zeros(L.shape[0], 2, device=L.device)
    sid[:, 0 if sensor == "modis" else 1] = 1.0
    g = torch.cat([sid, torch.stack([mean(z(cossza)), mean(z(vza, 60.0)), mean(ps[:, 0:1]), mean(ps[:, 1:2]), mf.mean((1, 2, 3))], 1)], 1)
    return x, g


class Stem(nn.Module):
    """Small per-sensor input layer: 2 masked convs, FiLM on tile-level conditions."""

    def __init__(self, w):
        super().__init__()
        self.c1 = MConv(N_FEAT, w)
        self.film = FiLM(N_COND, w)
        self.c2 = MConv(w, w)

    def forward(self, x, g, ctx):
        m = ctx.m.to(x.dtype)
        h = F.silu(self.c1(x, ctx))
        h = self.film(h, g.to(h.dtype)) * m
        return F.silu(self.c2(h, ctx)) * m


class Encoder(nn.Module):
    """Shared masked U-Net, 3 levels (H -> H/2 -> H/4 -> back); H, W divisible by 4. No sensor tag: the encoder must not know the sensor."""

    def __init__(self, w, k, skip_full=True):
        super().__init__()
        self.skip_full = skip_full
        self.d1 = MBlock(w, w)
        self.d2 = MBlock(w, 2 * w)
        self.d3 = MBlock(2 * w, 3 * w)
        self.u2 = MBlock(3 * w + 2 * w, 2 * w)
        self.u1 = MBlock(2 * w + (w if skip_full else 0), w)
        self.out = MConv(w, k, 1)

    def forward(self, h, ctx):
        c1, c2, c3 = ctx, ctx.down(), ctx.down().down()
        a = self.d1(h, c1)
        b = self.d2(mpool(a, c1), c2)
        c = self.d3(mpool(b, c2), c3)
        b = self.u2(torch.cat([mup(c, c2), b], 1), c2)
        up = mup(b, c1)
        a = self.u1(torch.cat([up, a], 1) if self.skip_full else up, c1)
        return self.out(a, c1)


class Conv3Encoder(nn.Module):
    """Baseline: three mask-aware convs (dilations 1,2,4), no down-sampling, same stem/decoder interface."""

    def __init__(self, w, k):
        super().__init__()
        self.c = nn.ModuleList([MConv(w, w, 3, d) for d in (1, 2, 4)])
        self.out = MConv(w, k, 1)

    def forward(self, h, ctx):
        for c in self.c:
            h = F.silu(c(h, ctx)) * ctx.m.to(h.dtype)
        return self.out(h, ctx)


class Decoder(nn.Module):
    """Shared decoder: latent -> per-cell (d4, d11, s4, s11 [, alloc logit])."""

    def __init__(self, k, w, n_out):
        super().__init__()
        self.b1 = MBlock(k, w)
        self.b2 = MBlock(w, w, dil=2)
        self.head = MConv(w, n_out, 1)
        nn.init.zeros_(self.head.conv.weight)

    def forward(self, z, ctx):
        return self.head(self.b2(self.b1(z, ctx), ctx), ctx)


class RawHarmonizer(nn.Module):
    def __init__(self, cfg: NetConfig | None = None):
        super().__init__()
        self.cfg = cfg or NetConfig()
        self.channels_last = False
        c = self.cfg
        self.stems = nn.ModuleDict({"modis": Stem(c.width), "viirs": Stem(c.width)})
        assert c.arch in ("unet", "unet_nobottleneck_skip", "conv3")
        self.encoder = Conv3Encoder(c.width, c.latent) if c.arch == "conv3" else Encoder(c.width, c.latent, c.arch == "unet")
        # head channels: [0:nb] background correction, [nb:2nb] excess log-ratio (nb = 2 bands, or 1 with out4), then [alloc logit], [hurdle logit, log amount]
        nb = c.n_out
        assert not (c.out4 and (c.alloc_head or c.hurdle_head or c.tie11)), "out4 supports only the free per-cell excess (no allocation / hurdle / tied 11 um)"
        self.i_alloc = 2 * nb if c.alloc_head else None
        self.i_hurdle = (2 * nb + int(c.alloc_head)) if c.hurdle_head else None
        self.decoder = Decoder(c.latent, c.width, 2 * nb + int(c.alloc_head) + 2 * int(c.hurdle_head))
        self.rho_gain = nn.Parameter(torch.ones(nb)) if c.rho_skip else None
        self.sun_k = nn.Parameter(torch.zeros(nb)) if c.sun_term else None
        with torch.no_grad():
            self.decoder.head.bias[nb:2 * nb] = c.s_bias
            if self.i_hurdle is not None:
                self.decoder.head.bias[self.i_hurdle] = -2.0  # P(fire) starts low

    def forward(self, x: torch.Tensor, in_mask: torch.Tensor, g: torch.Tensor, anchor_log: torch.Tensor, n_modis: int,
                sun: torch.Tensor | None = None):
        """x [N,N_FEAT,H,W] and g [N,N_COND]: the first n_modis rows are MODIS-sourced passes, the rest VIIRS-sourced.
        in_mask [N,H,W] bool = the common mask (network domain: nothing outside it ever enters; hidden cells are inside it and
        arrive zeroed with a visible-flag channel), anchor_log [N,nb,H,W] ln background anchor in the VIIRS band (nb = 2, or 1 with out4).
        -> dict(logB [N,nb,H,W] fp32, s [N,nb,H,W] fp32 >= 0, alloc [N,H,W] fp32 | None)."""
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        ctx = MaskCtx(in_mask[:, None].to(x.dtype))
        hs = []
        for name, sl in (("modis", slice(0, n_modis)), ("viirs", slice(n_modis, None))):
            if x[sl].shape[0]:
                sub = MaskCtx(ctx.m[sl])
                hs.append(self.stems[name](x[sl], g[sl], sub))
        h = torch.cat(hs, 0)
        o = self.decoder(self.encoder(h, ctx), ctx).float()
        c = self.cfg
        nb = c.n_out
        d = o[:, 0:nb]
        k = c.delta_lowres
        if k > 1:  # masked block average -> background correction varies only at 1/k resolution
            m = in_mask[:, None].float()
            d = F.interpolate(F.avg_pool2d(d * m, k) / F.avg_pool2d(m, k).clamp(min=1e-6), scale_factor=k, mode="nearest") * m
        logB = anchor_log + c.delta_max * torch.tanh(d)
        if self.sun_k is not None and sun is not None:
            logB = logB + 0.5 * torch.tanh(self.sun_k)[None, :, None, None] * sun.clamp(min=0.0)[:, None]
        s = F.softplus(o[:, nb:2 * nb])
        if self.rho_gain is not None:
            s = s + self.rho_gain[None, :, None, None] * (x[:, 2:2 + nb].float() * 2.0).clamp(min=0.0)  # rho channels of build_features (undo /2); with out4 only the 4 um one
        s = s.clamp(max=c.s_max)
        out = {"logB": logB, "s": s, "alloc": o[:, self.i_alloc] if self.i_alloc is not None else None, "hurdle": None, "amount": None}
        if self.i_hurdle is not None:
            out["hurdle"], out["amount"] = o[:, self.i_hurdle], o[:, self.i_hurdle + 1]
        return out
