import pytest
import torch

from nightrad.raw.losses import LossConfig, alloc_loss, apply_allocation, classes, hurdle_loss, huber, term_loss
from nightrad.raw.mask import Footprints


def test_huber_values():
    d = 0.1
    assert float(huber(torch.tensor(0.05), d)) == pytest.approx(0.5 * 0.05**2 / d)
    assert float(huber(torch.tensor(-0.3), d)) == pytest.approx(0.3 - 0.05)
    assert float(huber(torch.tensor(0.0), d)) == 0.0


def _args(b=1, h=6, w=6):
    mask = torch.ones(b, h, w, dtype=torch.bool)
    tgt = torch.zeros(b, 2, h, w)
    rho = torch.zeros(b, h, w)
    return mask, tgt, rho


def test_saturated_target_is_one_sided():
    cfg = LossConfig()
    mask, tgt, rho = _args()
    sat = torch.ones_like(mask)
    over = torch.full((1, 2, 6, 6), 0.0); over[:, 0] = 0.5   # prediction above a saturated target: free at 4 um
    under = torch.full((1, 2, 6, 6), 0.0); under[:, 0] = -0.5  # below: penalised
    l_over, _ = term_loss(over, tgt, mask, torch.ones(1, 6, 6), sat, cfg, rho)
    l_under, _ = term_loss(under, tgt, mask, torch.ones(1, 6, 6), sat, cfg, rho)
    l_unsat, _ = term_loss(over, tgt, mask, torch.ones(1, 6, 6), ~sat, cfg, rho)
    assert float(l_over) == 0.0 and float(l_under) > 0.0 and float(l_unsat) > 0.0


def test_loss_ignores_cells_outside_mask_even_if_nan():
    cfg = LossConfig()
    mask, tgt, rho = _args()
    mask[:, :3] = False
    pred = torch.zeros(1, 2, 6, 6, requires_grad=True)
    tgt2 = tgt.clone(); tgt2[:, :, :3] = float("nan")
    loss, _ = term_loss(pred + 0.2, tgt2, mask, torch.ones(1, 6, 6), torch.zeros_like(mask), cfg, rho)
    loss.backward()
    assert torch.isfinite(loss) and pred.grad[:, :, :3].abs().sum() == 0


def test_fire_weighting_and_class_balance_and_shares():
    mask, tgt, rho = _args(h=10, w=10)
    rho[0, 5, 5] = 1.0  # one fire-level cell
    pred = torch.zeros(1, 2, 10, 10); pred[:, :, 5, 5] = 0.3; pred[:, :, 0, 0] = 0.3  # equal error on a fire and on a background cell
    w = torch.ones(1, 10, 10)
    plain, st = term_loss(pred, tgt, mask, w, torch.zeros_like(mask), LossConfig(class_balance=False), rho)
    bal, stb = term_loss(pred, tgt, mask, w, torch.zeros_like(mask), LossConfig(class_balance=True), rho)
    # plain: the fire cell weighs 1+4*1 = 5x -> fire share of the objective is 5x the background cell's
    assert float(st["share_fire"]) == pytest.approx(5 / 6, rel=1e-4) and float(st["rawshare_fire"]) == pytest.approx(0.5, rel=1e-4)
    # balanced: each class gets its fixed share regardless of how many cells it has
    bg, near, fire = classes(rho, mask, LossConfig())
    assert fire.sum() == 1 and near.sum() == 24 and bg.sum() == 75
    assert float(stb["share_fire"]) > 0.5 and float(stb["share_fire"]) + float(stb["share_bg"]) + float(stb["share_near"]) == pytest.approx(1.0, abs=1e-5)
    only_bg = pred.clone(); only_bg[:, :, 5, 5] = 0
    l_nofire, _ = term_loss(only_bg, tgt, mask, w, torch.zeros_like(mask), LossConfig(class_balance=True), rho)
    assert float(l_nofire) < float(bal)


def test_allocation_conserves_footprint_mass_and_likelihood_beats_uniform():
    torch.manual_seed(0)
    pid = (torch.arange(8)[:, None] // 4 * 2 + torch.arange(8)[None, :] // 4).expand(2, 8, 8).long()
    fp = Footprints(pid)
    E = torch.rand(2, 2, 8, 8)
    logit = torch.randn(2, 8, 8)
    A = apply_allocation(E, logit, fp)
    assert torch.allclose(fp.total(A), fp.total(E), rtol=1e-5, atol=1e-6)  # energy conserved per footprint (P5 as a unit test, 1e-4)
    ok = torch.ones(2, 8, 8, dtype=torch.bool)
    e_meas = torch.rand(2, 8, 8) ** 4
    # a logit that equals ln(measured share) must beat the uniform baseline and a random one
    good, uni = alloc_loss(torch.log(e_meas + 1e-9), e_meas, fp, ok)
    rnd, _ = alloc_loss(logit, e_meas, fp, ok)
    assert float(good) < float(uni) - 0.05 and float(good) < float(rnd)
    flat, uni2 = alloc_loss(torch.zeros(2, 8, 8), e_meas, fp, ok)
    assert float(flat) == pytest.approx(float(uni2), rel=1e-4)


def test_hurdle_loss_rewards_correct_probability_and_amount():
    pid = (torch.arange(8)[:, None] // 4 * 2 + torch.arange(8)[None, :] // 4).expand(1, 8, 8).long()
    fp = Footprints(pid)
    ok = torch.ones(1, 8, 8, dtype=torch.bool)
    e = torch.zeros(1, 8, 8); e[0, :4, :4] = 0.4
    pos = e > 0
    amount_true = torch.full((1, 8, 8), float(torch.log(torch.tensor(0.4))))
    h_good = torch.where(pos, torch.tensor(6.0), torch.tensor(-6.0))
    cfg = LossConfig()
    good, _ = hurdle_loss(h_good, amount_true, e, fp, ok, pos, cfg)
    bad, _ = hurdle_loss(-h_good, amount_true - 3, e, fp, ok, pos, cfg)
    assert float(good) < 0.05 < float(bad)
