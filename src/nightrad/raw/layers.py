"""Mask-aware layers. Choice: PARTIAL (normalised) convolution + masked GroupNorm + masked pooling, with features forced to 0
outside the mask after every layer. Consequence (tested exactly): a value outside the mask cannot influence any output inside it,
and zero padding/mask edges do not leave an edge signature (each conv is re-normalised by its in-mask tap count).
A plain zero-padded convolution would leak masked values, so every convolution here is the partial one."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    """Feature-wise affine modulation by a conditioning vector; zero-initialised so it starts as the identity."""

    def __init__(self, cond, ch):
        super().__init__()
        self.lin = nn.Linear(cond, 2 * ch)
        nn.init.zeros_(self.lin.weight); nn.init.zeros_(self.lin.bias)

    def forward(self, x, c):
        gb = self.lin(c)[:, :, None, None]
        g, b = gb.chunk(2, 1)
        return x * (1 + g) + b


class MaskCtx:
    """Mask [B,1,H,W] float 0/1 plus cached per-(k,dilation) renormalisation maps and the down-sampled context."""

    def __init__(self, m: torch.Tensor):
        self.m = m
        self._scale: dict = {}
        self._down: "MaskCtx | None" = None

    def count(self) -> torch.Tensor:
        if not hasattr(self, "_cnt"):
            self._cnt = self.m.float().sum((1, 2, 3))
        return self._cnt

    def scale_m(self, k: int, d: int, dtype) -> torch.Tensor:
        key = ("sm", k, d, dtype)
        if key not in self._scale:
            self._scale[key] = (self.scale(k, d) * self.m).to(dtype)
        return self._scale[key]

    def scale(self, k: int, d: int) -> torch.Tensor:
        key = (k, d)
        if key not in self._scale:
            ones = torch.ones(1, 1, k, k, device=self.m.device, dtype=self.m.dtype)
            cnt = F.conv2d(self.m, ones, padding=d * (k // 2), dilation=d)
            self._scale[key] = (k * k) / cnt.clamp(min=1.0)
        return self._scale[key]

    def down(self) -> "MaskCtx":
        if self._down is None:
            self._down = MaskCtx(F.max_pool2d(self.m, 2))
        return self._down


class MConv(nn.Module):
    """Partial convolution. INVARIANT (kept by every layer here and by build_features): inputs are exactly 0 outside the mask, so the
    input needs no re-masking; output = conv(x) * (k^2 / in-mask taps) * mask, the scale and mask fused into one cached map.
    bias=False for convs followed by GroupNorm (the norm cancels it) saves a full pass over the tensor."""

    def __init__(self, cin, cout, k=3, dil=1, bias=True):
        super().__init__()
        self.k, self.dil = k, dil
        self.conv = nn.Conv2d(cin, cout, k, padding=dil * (k // 2), dilation=dil, bias=False)
        self.bias = nn.Parameter(torch.zeros(cout)) if bias else None

    def forward(self, x, ctx: MaskCtx):
        y = self.conv(x) * ctx.scale_m(self.k, self.dil, x.dtype)
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)[None, :, None, None] * ctx.m.to(y.dtype)
        return y


class MGroupNorm(nn.Module):
    """GroupNorm with statistics over in-mask cells only. Input must be 0 outside the mask (invariant); sums are taken in fp32 without
    materialising an fp32 copy; output = (x*a + c) * mask."""

    def __init__(self, c, groups=8, eps=1e-5):
        super().__init__()
        self.g = groups if c % groups == 0 else 1
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))

    def forward(self, x, ctx: MaskCtx):
        b, c, h, w = x.shape
        cg = c // self.g
        n = (ctx.count() * cg).clamp(min=1.0)  # [B] in-mask cells x channels per group
        s1 = x.sum((2, 3), dtype=torch.float32).view(b, self.g, cg).sum(2)
        s2 = torch.linalg.vector_norm(x, dim=(2, 3), dtype=torch.float32).pow(2).view(b, self.g, cg).sum(2)
        mean = s1 / n[:, None]
        rstd = torch.rsqrt((s2 / n[:, None] - mean * mean).clamp(min=0) + self.eps)
        mean_c, rstd_c = mean.repeat_interleave(cg, 1), rstd.repeat_interleave(cg, 1)  # [B,C]
        a = rstd_c * self.weight[None]
        cst = self.bias[None] - mean_c * a
        y = torch.addcmul(cst.to(x.dtype)[:, :, None, None], x, a.to(x.dtype)[:, :, None, None])
        return y * ctx.m.to(x.dtype)


class MBlock(nn.Module):
    """conv-GN-SiLU [FiLM] conv-GN-SiLU, all mask-aware."""

    def __init__(self, cin, cout, cond=None, dil=1):
        super().__init__()
        self.c1, self.n1 = MConv(cin, cout, 3, dil, bias=False), MGroupNorm(cout)
        self.c2, self.n2 = MConv(cout, cout, 3, dil, bias=False), MGroupNorm(cout)
        self.film = FiLM(cond, cout) if cond else None

    def forward(self, x, ctx: MaskCtx, cond=None):
        x = F.silu(self.n1(self.c1(x, ctx), ctx))
        if self.film is not None:
            x = self.film(x, cond.to(x.dtype)) * ctx.m.to(x.dtype)
        return F.silu(self.n2(self.c2(x, ctx), ctx)) * ctx.m.to(x.dtype)


def mpool(x, ctx: MaskCtx):
    """Masked 2x2 average: only in-mask children contribute."""
    m = ctx.m.to(x.dtype)
    num = F.avg_pool2d(x * m, 2)
    den = F.avg_pool2d(m, 2)
    return (num / den.clamp(min=1e-6)) * ctx.down().m.to(x.dtype)


def mup(x, ctx_fine: MaskCtx):
    return F.interpolate(x, scale_factor=2, mode="nearest") * ctx_fine.m.to(x.dtype)
