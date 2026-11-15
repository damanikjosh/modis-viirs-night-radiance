"""descriptive measures (FSS, precision-recall / AP / CSI, right hottest cell, amount skill): every measure on synthetic fields with a KNOWN answer, the guard, determinism, and a
cross-check against the stored protocol code (protocol_eval.collect) on synthetic tiles."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from nightrad.raw import descriptive as D

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPLIT = os.path.join(ROOT, "configs", "nf2_split.json")


# ------------------------------------------------------------------ FSS
def _fss(obs, fc, k, valid=None):
    obs, fc = torch.as_tensor(obs)[None], torch.as_tensor(fc)[None]
    valid = torch.ones_like(obs) if valid is None else torch.as_tensor(valid)[None]
    num, den, _ = D.fss_terms(obs, fc, valid, k)
    return D.fss_value(float(num.sum()), float(den.sum()))


def _blob(h, w, cells):
    a = np.zeros((h, w), bool)
    for r, c in cells:
        a[r, c] = True
    return a


def test_fss_perfect_field_is_one_at_every_scale():
    rng = np.random.default_rng(0); f = rng.random((41, 41)) < 0.1
    for k in D.SCALES:
        assert _fss(f, f, k) == pytest.approx(1.0, abs=1e-9)


def test_fss_displaced_single_cell_matches_the_closed_form():
    # one fire cell displaced by d cells along a row: windows of edge n overlap in n*(n-d) cells -> FSS = (n-d)/n for d < n, else 0
    obs, fc = _blob(61, 61, [(30, 30)]), _blob(61, 61, [(30, 33)])
    assert _fss(obs, fc, 1) == pytest.approx(0.0, abs=1e-9)
    assert _fss(obs, fc, 3) == pytest.approx(0.0, abs=1e-9)            # d = 3 is not < n = 3
    assert _fss(obs, fc, 5) == pytest.approx(2 / 5, abs=1e-6)
    assert _fss(obs, fc, 9) == pytest.approx(6 / 9, abs=1e-6)
    assert _fss(obs, fc, 27) == pytest.approx(24 / 27, abs=1e-6)


def test_fss_smeared_field_is_lower_at_fine_scale_and_equal_at_coarse_scale():
    c = 20
    sharp_obs = np.zeros((41, 41), bool); sharp_obs[c - 1:c + 2, c - 1:c + 2] = True           # 9 cells in a 3x3 clump
    lattice = np.zeros((41, 41), bool)
    for dy in (-3, 0, 3):
        for dx in (-3, 0, 3):
            lattice[c + dy, c + dx] = True                                                      # the same 9 cells spread over 9x9
    assert _fss(sharp_obs, sharp_obs.copy(), 1) == pytest.approx(1.0)                          # correct and sharp: perfect at the finest scale
    assert _fss(sharp_obs, lattice, 1) == pytest.approx(1 - 16 / 18, abs=1e-9)                  # smeared: only the centre cell overlaps
    s = [_fss(sharp_obs, lattice, k) for k in (1, 3, 5, 9, 15, 27)]
    assert all(a <= b + 1e-9 for a, b in zip(s, s[1:])) and s[0] < 0.2 and s[-1] > 0.9         # rises with scale
    assert _fss(sharp_obs, lattice, 121) == pytest.approx(1.0, abs=1e-9)                       # window covers the domain: equal fire amount -> equal fractions


def test_fss_amount_error_is_penalised_at_all_scales_and_empty_is_undefined():
    obs = _blob(21, 21, [(10, 10)]); empty = np.zeros((21, 21), bool)
    assert _fss(obs, empty, 9) == pytest.approx(0.0, abs=1e-9)    # a model that flags nothing scores 0, never 1
    assert np.isnan(_fss(empty, empty, 9))                        # no fire in either field: undefined (not counted)


def test_fss_invalid_cells_do_not_count_and_even_windows_are_refused():
    obs = _blob(21, 21, [(10, 10), (2, 2)]); fc = _blob(21, 21, [(10, 10), (18, 18)])
    valid = np.ones((21, 21), bool); valid[:6, :6] = False; valid[15:, 15:] = False   # the two disagreeing cells sit in invalid corners
    assert _fss(obs, fc, 1, valid) == pytest.approx(1.0)
    assert _fss(obs, fc, 1) < 1.0
    with pytest.raises(AssertionError):
        _fss(obs, fc, 4)


# ------------------------------------------------------------------ AP / PR / CSI
def test_ap_known_small_example_and_perfect_and_worst():
    assert D.average_precision(np.array([0.9, 0.8, 0.7, 0.6]), np.array([1, 0, 1, 0])) == pytest.approx(0.5 * 1.0 + 0.5 * 2 / 3)
    assert D.average_precision(np.array([4, 3, 2, 1.0]), np.array([1, 1, 0, 0])) == pytest.approx(1.0)
    assert D.average_precision(np.array([1, 2, 3, 4.0]), np.array([1, 1, 0, 0])) == pytest.approx(0.5 * (1 / 3) + 0.5 * 0.5)


def test_ap_of_a_random_scorer_equals_the_base_rate():
    rng = np.random.default_rng(1); lab = rng.random(400_000) < 0.05
    assert abs(D.average_precision(rng.random(lab.size), lab) - lab.mean()) < 0.004


def test_ap_of_a_constant_scorer_is_exactly_the_base_rate_ties_are_one_threshold():
    lab = np.r_[np.ones(7, bool), np.zeros(93, bool)]
    assert D.average_precision(np.full(100, 0.3), lab) == pytest.approx(0.07)
    assert np.isnan(D.average_precision(np.arange(5.0), np.zeros(5, bool)))   # no positive: undefined


def test_histogram_ap_matches_exact_ap_within_the_bin_width():
    rng = np.random.default_rng(2); n = 200_000
    lab = rng.random(n) < 0.03
    score = np.where(lab, np.exp(rng.normal(-3, 1.2, n)), np.exp(rng.normal(-5, 1.5, n))); score[rng.random(n) < 0.5] = 0.0   # half of the cells exactly 0, as in real excess maps
    edges = D.hist_edges().numpy(); b = np.searchsorted(edges, score, side="right")
    pos, neg = np.bincount(b[lab], minlength=D.NB), np.bincount(b[~lab], minlength=D.NB)
    assert abs(D.curve_from_counts(pos, neg)[2] - D.average_precision(score, lab)) < 2e-3


def test_csi_counts():
    assert D.csi_from_counts(6, 2, 2) == pytest.approx(0.6)
    assert D.csi_from_counts(0, 3, 4) == 0.0
    assert np.isnan(D.csi_from_counts(0, 0, 0))


def test_thinned_curve_keeps_ends_and_is_only_display():
    rng = np.random.default_rng(3); lab = rng.random(20_000) < 0.1; s = rng.random(20_000) + 0.5 * lab
    p, r, ap, base = D.pr_curve(s, lab); cr, cp = D.thin_curve(p, r, 100)
    assert len(cr) <= 101 and cr[-1] == pytest.approx(1.0) and ap == pytest.approx(D.average_precision(s, lab))


# ------------------------------------------------------------------ right hottest cell
def _grid(h=6, w=6, ps=3):
    pid = np.array([[(r // ps) * (w // ps) + (c // ps) for c in range(w)] for r in range(h)]).reshape(1, -1)
    return torch.tensor(pid), h * w + 1


def test_hottest_cell_perfect_shifted_and_uniform():
    pid, n = _grid(); rng = np.random.default_rng(0)
    T = torch.tensor(rng.random((1, 36)) + 0.1)
    ex, wi, ce, cw, tied = D.hottest_cell(T.clone(), T, pid, n, 6, 6)
    assert torch.allclose(ex[0, :4], torch.ones(4)) and torch.allclose(wi[0, :4], torch.ones(4))
    # peak at (1,1) in pixel 0; the method puts its peak one cell to the right at (1,2): not the right cell, but within one cell
    T2 = torch.full((1, 36), 0.1, dtype=torch.float64); T2[0, 1 * 6 + 1] = 5.0
    V2 = torch.full((1, 36), 0.1, dtype=torch.float64); V2[0, 1 * 6 + 2] = 5.0
    ex, wi, *_ = D.hottest_cell(V2, T2, pid, n, 6, 6)
    assert ex[0, 0] == 0.0 and wi[0, 0] == 1.0
    # peak two cells away (1,3) lies in pixel 1: pixel 0's own maximum is then elsewhere; take a method with its peak at (2,... ) distance 2 inside pixel: truth (0,0), method (2,2)
    T3 = torch.full((1, 36), 0.1, dtype=torch.float64); T3[0, 0] = 5.0
    V3 = torch.full((1, 36), 0.1, dtype=torch.float64); V3[0, 2 * 6 + 2] = 5.0
    ex, wi, *_ = D.hottest_cell(V3, T3, pid, n, 6, 6)
    assert ex[0, 0] == 0.0 and wi[0, 0] == 0.0
    # perfectly flat prediction (a uniform spread): the expected credit is exactly the chance level, for the exact and the within-one question
    ex, wi, ce, cw, tied = D.hottest_cell(torch.ones(1, 36, dtype=torch.float64), T, pid, n, 6, 6)
    assert torch.allclose(ex[0, :4], ce[0, :4]) and torch.allclose(ce[0, :4], torch.full((4,), 1 / 9, dtype=ce.dtype))
    assert torch.allclose(wi[0, :4], cw[0, :4])


def test_hottest_cell_random_predictor_is_at_chance():
    pid, n = _grid(120, 120, 3); rng = np.random.default_rng(4)
    T = torch.tensor(rng.random((1, 14400))); V = torch.tensor(rng.random((1, 14400)))
    ex, wi, ce, cw, tied = D.hottest_cell(V, T, pid, n, 120, 120)
    m = slice(0, 1600)
    assert abs(float(ex[0, m].mean()) - 1 / 9) < 0.03 and abs(float(wi[0, m].mean()) - float(cw[0, m].mean())) < 0.04
    assert float(cw[0, m].mean()) > float(ce[0, m].mean())   # within one cell is an easier question than the exact cell


def test_hottest_cell_tied_truth_is_reported_and_every_tied_cell_counts_as_hottest():
    pid, n = _grid(); T = torch.ones(1, 36, dtype=torch.float64)
    ex, wi, ce, cw, tied = D.hottest_cell(torch.rand(1, 36, dtype=torch.float64), T, pid, n, 6, 6)
    assert float(tied[0, :4].min()) == 1.0 and torch.allclose(ex[0, :4], torch.ones(4)) and torch.allclose(ce[0, :4], torch.ones(4))   # every cell is a hottest cell: always right, chance 1
    # two tied hottest cells in a 9-cell pixel: a random pick is right with probability 2/9 (= chance); a method whose maximum is one of them is right
    T2 = torch.full((1, 36), 0.1, dtype=torch.float64); T2[0, 0] = 5.0; T2[0, 2 * 6 + 2] = 5.0
    V2 = torch.full((1, 36), 0.1, dtype=torch.float64); V2[0, 0] = 9.0
    ex, wi, ce, cw, tied = D.hottest_cell(V2, T2, pid, n, 6, 6)
    assert ex[0, 0] == 1.0 and float(ce[0, 0]) == pytest.approx(2 / 9) and tied[0, 0] == 1.0
    V3 = torch.full((1, 36), 0.1, dtype=torch.float64); V3[0, 1 * 6 + 1] = 9.0   # centre cell: not hottest, but adjacent (within one) to both
    ex, wi, *_ = D.hottest_cell(V3, T2, pid, n, 6, 6)
    assert ex[0, 0] == 0.0 and wi[0, 0] == 1.0


# ------------------------------------------------------------------ amount skill
def test_amount_skill_known_values_and_distribution_over_pairs():
    rng = np.random.default_rng(5); pairs = np.repeat(np.arange(6), 30)
    meas = 50.0 + 50 * rng.random(pairs.size)
    even = meas * np.exp(1.0); model = meas * np.exp(0.5); worse = meas * np.exp(2.0)
    items = {"pair": pairs, "meas": meas, "pred_uniform": even, "pred_model": model, "pred_excess_only": even, "pred_copy_raw": worse}
    r = D.amount_skill(items, methods=("model", "uniform", "copy_raw"), refs=("uniform",), n_boot=200)
    s = r["skill"]["uniform"]
    assert s["model"]["mean"] == pytest.approx(0.5, abs=0.01) and s["model"]["share_pairs_above_0"] == 1.0 and s["model"]["n_pairs"] == 6
    assert s["uniform"]["mean"] == pytest.approx(0.0, abs=1e-9)
    assert s["copy_raw"]["mean"] == pytest.approx(-1.0, abs=0.02)
    assert len(s["model"]["per_pair"]) == 6 and s["model"]["min"] <= s["model"]["median"] <= s["model"]["max"] and s["model"]["lo"] <= s["model"]["mean"] <= s["model"]["hi"]


def test_amount_skill_leaves_out_small_pairs_and_is_deterministic():
    pairs = np.r_[np.zeros(30), np.ones(5)].astype(int); meas = np.full(35, 40.0)
    items = {"pair": pairs, "meas": meas, "pred_uniform": meas * 2, "pred_model": meas * 1.2}
    a = D.amount_skill(items, methods=("model", "uniform"), refs=("uniform",), n_boot=100); b = D.amount_skill(items, methods=("model", "uniform"), refs=("uniform",), n_boot=100)
    assert a["pairs"] == [0] and a == b


# ------------------------------------------------------------------ guard: no REPORT tile can reach a measure
def _make_pack(d, names, episodes):
    os.makedirs(d, exist_ok=True); n = len(names)
    json.dump({"n": n, "keys": ["eval_episode"]}, open(os.path.join(d, "meta.json"), "w"))
    ep = {e: i for i, e in enumerate(sorted(set(episodes)))}
    np.save(os.path.join(d, "eval_episode.npy"), np.array([ep[e] for e in episodes], np.int32)); json.dump(ep, open(os.path.join(d, "episodes.json"), "w"))
    json.dump(names, open(os.path.join(d, "names.json"), "w")); return str(d)


def _lists():
    t = json.load(open(SPLIT))["tiles"]; return t["select"], t["report"]


def _eps(tmp):
    f = tmp / "eps.json"; json.dump({"select": ["ES"], "report": ["ER"]}, open(f, "w")); return str(f)


def test_select_set_drops_report_tiles_of_a_mixed_pack(tmp_path):
    sel, rep = _lists()
    d = _make_pack(tmp_path / "p", sel[:3] + rep[:2], ["ES"] * 3 + ["ER"] * 2)
    idx, guard = D.select_tile_indices(d, "select", _eps(tmp_path), SPLIT)
    assert idx == [0, 1, 2] and guard["checked"] and guard["tiles_checked"] == 3 and guard["report_hits"] == 0 and guard["split_pinned"]
    names = json.load(open(os.path.join(d, "names.json")))
    assert not ({names[i] for i in idx} & set(rep))


def test_a_report_id_hidden_behind_a_select_episode_is_refused_by_the_guard(tmp_path):
    sel, rep = _lists()
    d = _make_pack(tmp_path / "p", sel[:2] + rep[:1], ["ES"] * 3)           # a REPORT tile id wrongly labelled as a SELECT episode
    with pytest.raises(SystemExit) as e:
        D.select_tile_indices(d, "select", _eps(tmp_path), SPLIT)
    assert "REPORT" in str(e.value)


def test_retired_ids_are_refused(tmp_path):
    sel, rep = _lists(); ret = sorted(json.load(open(os.path.join(ROOT, "configs", "report_retired.json")))["retired_tile_ids"])
    d = _make_pack(tmp_path / "p", sel[:2] + ret[:1], ["ES"] * 3)
    with pytest.raises(SystemExit) as e:
        D.select_tile_indices(d, "select", _eps(tmp_path), SPLIT)
    assert "RETIRED" in str(e.value) or "REPORT" in str(e.value)


@pytest.mark.parametrize("bad", ["report", "val2all", "test", "REPORT"])
def test_report_and_other_sets_cannot_be_requested(tmp_path, bad):
    sel, rep = _lists(); d = _make_pack(tmp_path / "p", sel[:1], ["ES"])
    with pytest.raises(ValueError):
        D.select_tile_indices(d, bad, _eps(tmp_path), SPLIT)


def test_cli_refuses_report_set():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "descriptive_score.py"), "--set", "report", "--data", "x", "--runs-dir", "x", "--out", "x", "--git-sha", "abcdef1"], capture_output=True, text=True, cwd=ROOT,
                       env={**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")})
    assert r.returncode != 0 and "invalid choice" in r.stderr


def test_selection_equals_the_one_scripts_score_run_makes(tmp_path):
    """Same pack, same split: the tiles my selection hands to the measures are the tiles scripts/score_run.py hands to its guard (it prints the guard record before loading any tile)."""
    sel, rep = _lists(); d = _make_pack(tmp_path / "p", sel[:3] + rep[:2], ["ES"] * 3 + ["ER"] * 2); eps = _eps(tmp_path)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "score_run.py"), "--data", d, "--train", d, "--runs-dir", str(tmp_path), "--out", str(tmp_path / "o"), "--set", "select", "--git-sha", "abcdef1234", "--split", eps,
                        "--nf2-split", SPLIT, "--names", "none"], cwd=ROOT, env={**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")}, capture_output=True, text=True, timeout=300)
    idx, guard = D.select_tile_indices(d, "select", eps, SPLIT)
    assert f"'tiles_checked': {guard['tiles_checked']}" in r.stdout and guard["tiles_checked"] == len(idx) == 3


# ------------------------------------------------------------------ cross-check against the stored protocol code on synthetic tiles
@pytest.fixture(scope="module")
def synth_run():
    from nightrad.raw import protocol_eval as PE
    from nightrad.raw.core import RunConfig
    from nightrad.raw.net import NetConfig, RawHarmonizer
    from nightrad.raw.synth import SyntheticTiles

    class DS(SyntheticTiles):
        def __getitem__(self, i):
            t = dict(super().__getitem__(i)); t["eval_pair_id"] = torch.tensor(i % 3, dtype=torch.int32); return t
    torch.manual_seed(0)
    cfg = RunConfig(net=NetConfig(width=8, latent=4), daynight="both"); model = RawHarmonizer(cfg.net).eval()
    tiles = PE.load_all(DS(18, 7, 48, 48, cache=True))
    return PE, cfg, model, tiles


def test_flag_counts_block_hit_rate_and_auc_reproduce_protocol_eval(synth_run):
    PE, cfg, model, tiles = synth_run
    dev = torch.device("cpu"); const = -0.3
    res = PE.collect(model, tiles, cfg, dev, const, bs=6); ref = PE.summarize(res)
    acc = D.Acc(); D.collect(model, tiles, cfg, dev, const, acc, bs=6); out = D.summarize(acc)
    seen = 0
    for m in ("model", "model_off", "uniform", "copy_planck", "copy_raw"):
        r, op = ref["rule"][m], out["blocks"][m]["operating_point"]
        assert op["flagged_cells"] == r["flagged_cells_pred"] and op["measured_cells"] == r["flagged_cells_meas"]
        assert op["recall"] == r["block_hit_rate"] or (op["recall"] is None and r["block_hit_rate"] is None)
        far = None if (op["tp"] + op["fp"]) == 0 else op["fp"] / (op["tp"] + op["fp"])
        assert far == r["block_false_alarm_ratio"]
        seen += op["tp"] + op["fn"] + op["flagged_cells"]
    for m in ("model", "uniform", "copy_planck", "excess_only"):   # block scores identical -> identical per-pair AUC
        pm = "oracle_uniform" if m == "excess_only" else m
        for p, (s, l) in res["blocks"][pm].items():
            s2 = np.concatenate(acc.blocks[(m, p)][0]); l2 = np.concatenate(acc.blocks[(m, p)][1])
            assert np.array_equal(np.sort(s), np.sort(s2)) and l.sum() == l2.sum()
    assert seen > 0, "the synthetic tiles carried no fire at all: the cross-check would be empty"


def test_collect_is_deterministic_and_chunk_independent(synth_run):
    PE, cfg, model, tiles = synth_run
    dev = torch.device("cpu")
    a1 = D.Acc(); D.collect(model, tiles, cfg, dev, -0.3, a1, bs=6)
    a2 = D.Acc(); D.collect(model, tiles, cfg, dev, -0.3, a2, bs=6)
    half = lambda lo, hi: {k: v[lo:hi] for k, v in tiles.items()}
    a3 = D.Acc(); D.collect(model, half(0, 9), cfg, dev, -0.3, a3, bs=6); D.collect(model, half(9, 18), cfg, dev, -0.3, a3, bs=6)
    j = lambda a: json.dumps(D.summarize(a), sort_keys=True, default=float)
    assert j(a1) == j(a2)
    s1, s3 = json.loads(j(a1)), json.loads(j(a3))
    assert s1["blocks"]["model"]["operating_point"]["tp"] == s3["blocks"]["model"]["operating_point"]["tp"]
    for m in ("model", "uniform"):
        for k in D.SCALES:
            a, b = s1["fss"][m][str(k)]["pooled"], s3["fss"][m][str(k)]["pooled"]
            assert (a != a and b != b) or a == pytest.approx(b, abs=1e-9)


def test_simple_methods_alone_give_the_same_numbers_as_inside_a_model_run(synth_run):
    PE, cfg, model, tiles = synth_run
    dev = torch.device("cpu")
    a1 = D.Acc(); D.collect(model, tiles, cfg, dev, -0.3, a1, bs=6)
    a2 = D.Acc(); D.collect(None, tiles, cfg, dev, -0.3, a2, bs=6)
    s1, s2 = D.summarize(a1), D.summarize(a2)
    assert "model" not in s2["blocks"] and "model_off" not in s2["fss"]
    for m in D.SIMPLE:
        assert json.dumps(s1["blocks"][m], sort_keys=True, default=float) == json.dumps(s2["blocks"][m], sort_keys=True, default=float)
        assert json.dumps(s1["hottest"][m], sort_keys=True, default=float) == json.dumps(s2["hottest"][m], sort_keys=True, default=float)


def test_a_flat_spread_sits_at_chance_in_the_real_collector(synth_run):
    """'const' is flat inside every MODIS pixel only through the background map, so use the pure ratio of exact over chance on the 'uniform' method of the synthetic tiles where MODIS is constant per pixel."""
    PE, cfg, model, tiles = synth_run
    acc = D.Acc(); D.collect(model, tiles, cfg, torch.device("cpu"), -0.3, acc, bs=6); out = D.summarize(acc)
    h = out["hottest"]["copy_raw"]["fire_level"]     # copy of the MODIS radiance: constant inside each MODIS pixel -> every cell tied -> credit equals chance
    if h.get("n_pixels", 0) > 0:
        assert h["exact_pooled"] == pytest.approx(h["chance_exact"], abs=1e-9) and h["within1_pooled"] == pytest.approx(h["chance_within1"], abs=1e-9)
