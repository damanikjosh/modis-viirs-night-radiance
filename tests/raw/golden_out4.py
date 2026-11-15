"""Fixed-batch reference computation for the 4 um-only option tests. `compute(mode)` is run with the option OFF; its numbers were recorded
from the code BEFORE the option existed (tests/raw/golden_out4_off.json) and the test asserts the code after the change reproduces them bit for bit."""
import hashlib

import torch

from nightrad.raw.core import RunConfig, forward_loss, run_scenes, Prep
from nightrad.raw.losses import LossConfig
from nightrad.raw.net import NetConfig, RawHarmonizer
from nightrad.raw.schema import split_batch
from nightrad.raw.synth import SyntheticTiles


def fixed_batch(n=4):
    ds = SyntheticTiles(n, 0, 96, 96, cache=True)
    items = [ds[i] for i in range(n)]
    return {k: torch.stack([it[k] for it in items]) for k in items[0]}


def _h(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


def compute(mode: str, **net_kw):
    torch.manual_seed(0)
    cfg = RunConfig(net=NetConfig(width=8, latent=4, **net_kw), loss=LossConfig(mode=mode))
    model = RawHarmonizer(cfg.net)
    inp, _ = split_batch(fixed_batch(), torch.device("cpu"))
    gen = torch.Generator().manual_seed(1234)
    P = Prep(inp, cfg)
    with torch.no_grad():
        sc, _ = run_scenes(model, P, cfg, torch.Generator().manual_seed(1234))
    loss, logs = forward_loss(model, inp, cfg, torch.Generator().manual_seed(1234))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    opt.zero_grad(); loss.backward(); opt.step()
    loss2, _ = forward_loss(model, inp, cfg, torch.Generator().manual_seed(1234))
    s = sc[("modis", "clean")]
    return {"loss": float(loss), "loss_after_step": float(loss2), "logs": {k: float(v) for k, v in sorted(logs.items())},
            "B_hash": _h(s["B"]), "E_hash": _h(s["E"]), "n_params": sum(p.numel() for p in RawHarmonizer(cfg.net).parameters()), "torch": torch.__version__}
