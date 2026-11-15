"""the validation loss logged by diag.full_eval equals the training-path loss on the same batch (same hiding draw), and its terms sum to the total."""
import torch

from nightrad.raw import diag
from nightrad.raw.core import RunConfig, forward_loss
from nightrad.raw.losses import LossConfig
from nightrad.raw.net import NetConfig, RawHarmonizer
from nightrad.raw.schema import split_batch
from nightrad.raw.synth import SyntheticTiles


def _batch(n=4):
    ds = SyntheticTiles(n, 0, 96, 96, cache=True)
    items = [ds[i] for i in range(n)]
    return {k: torch.stack([it[k] for it in items]) for k in items[0]}


def _check(mode):
    cfg = RunConfig(net=NetConfig(width=8, latent=4), loss=LossConfig(mode=mode))
    torch.manual_seed(0)
    model = RawHarmonizer(cfg.net)
    b = _batch()
    model.train()
    inp, _ = split_batch(b, torch.device("cpu"))
    gen = torch.Generator().manual_seed(1234)
    lt, _ = forward_loss(model, inp, cfg, gen, None)
    fe = diag.full_eval_weighted(model, [b], cfg, torch.device("cpu"), None, seed=1234)
    assert abs(fe["val_total"] - float(lt)) <= 1e-6 * max(1.0, abs(float(lt))), (fe["val_total"], float(lt))
    assert abs(fe["terms_weighted_sum"] - fe["val_total"]) <= 1e-5 * max(1.0, fe["val_total"]), (fe["terms_weighted_sum"], fe["val_total"])
    assert model.training  # full_eval restores the mode


def test_parity_gamma():
    _check("gamma")


def test_parity_legacy():
    _check("legacy")


def test_epoch_generator_all_samplers():
    """Region-balanced / shuffled / fire-batch loaders all expose a seedable generator (the reference run crashed on the region-balanced loader)."""
    import numpy as np
    from nightrad.raw.data import make_loader, region_balanced_weights
    from nightrad.raw.train import epoch_generator
    ds = SyntheticTiles(8, 0, 96, 96, cache=True)
    w = region_balanced_weights(np.array([0, 0, 0, 0, 1, 1, 2, 2]))
    for ld in (make_loader(ds, 4, 0, True, 0, drop_last=True), make_loader(ds, 4, 0, True, 0, drop_last=True, sample_weights=w)):
        g = epoch_generator(ld)
        assert g is not None
        g.manual_seed(5); a = list(iter(ld.sampler if ld.batch_sampler is None else ld.batch_sampler))
        g.manual_seed(5); b = list(iter(ld.sampler if ld.batch_sampler is None else ld.batch_sampler))
        assert a == b


def test_bright_weight_and_head_lr():
    """bright_weight=0 leaves the gamma loss unchanged; >0 raises the share of MODIS-bright cells; head_lr_mult=1 is one group, >1 two groups covering all parameters once."""
    from nightrad.raw.train import make_param_groups
    b = _batch(); torch.manual_seed(0)
    cfg0 = RunConfig(net=NetConfig(width=8, latent=4), loss=LossConfig(mode="gamma"))
    model = RawHarmonizer(cfg0.net)
    inp, _ = split_batch(b, torch.device("cpu"))
    l0, _ = forward_loss(model, inp, cfg0, torch.Generator().manual_seed(1), None)
    cfg1 = RunConfig(net=NetConfig(width=8, latent=4), loss=LossConfig(mode="gamma", bright_weight=0.0))
    l1, _ = forward_loss(model, inp, cfg1, torch.Generator().manual_seed(1), None)
    assert float(l0) == float(l1)
    cfg2 = RunConfig(net=NetConfig(width=8, latent=4), loss=LossConfig(mode="gamma", bright_weight=100.0))
    l2, _ = forward_loss(model, inp, cfg2, torch.Generator().manual_seed(1), None)
    assert torch.isfinite(l2) and float(l2) != float(l1)
    g1, g2 = make_param_groups(model, 1e-4, 1.0), make_param_groups(model, 1e-4, 5.0)
    assert len(g1) == 1 and len(g2) == 2
    ids = [id(p) for grp in g2 for p in grp["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters())) and g2[1]["lr_mult"] == 5.0
