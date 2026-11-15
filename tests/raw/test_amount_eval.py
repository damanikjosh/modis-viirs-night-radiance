"""Amendment 4 (amount measure): metric code on synthetic items (tests written before the implementation)."""
import numpy as np

from nightrad.raw import amount_eval as AE


def _items(n_pairs=8, per=40, seed=0):
    rng = np.random.default_rng(seed)
    pair = np.repeat(np.arange(n_pairs), per)
    meas = np.exp(rng.normal(0, 1, pair.size))
    return pair, meas, rng


def test_log_error_and_ratio():
    e = AE.log_error(np.array([2.0, 0.5]), np.array([1.0, 1.0]), eps=0.0)
    assert np.allclose(e, [np.log(2), np.log(0.5)])
    assert np.isclose(AE.ratio_of_sums(np.array([2.0, 2.0]), np.array([1.0, 3.0])), 1.0)


def test_spearman():
    x = np.arange(10.0)
    assert np.isclose(AE.spearman(x, x ** 3), 1.0)
    assert np.isclose(AE.spearman(x, -x), -1.0)
    assert abs(AE.spearman(np.r_[1, 1, 1, 1.0], np.r_[1, 2, 3, 4.0])) < 1e-9 or np.isnan(AE.spearman(np.r_[1, 1, 1, 1.0], np.r_[1, 2, 3, 4.0]))


def test_calibration_perfect_and_biased():
    pair, meas, rng = _items()
    c = AE.calibration(meas, meas, pair, nbins=8, n_boot=200)
    assert all(abs(b["mean_pred"] - b["mean_meas"]) < 1e-9 for b in c)
    assert sum(b["n"] for b in c) == meas.size
    c2 = AE.calibration(2 * meas, meas, pair, nbins=8, n_boot=200)
    assert all(b["mean_pred"] > 1.9 * b["mean_meas"] for b in c2)
    assert all(b["lo"] <= b["mean_meas"] <= b["hi"] for b in c2)


def test_decision_model_better_than_biased_baselines():
    pair, meas, rng = _items()
    preds = {"model": meas * np.exp(rng.normal(0, 0.1, meas.size)), "spread": meas * 2.5 * np.exp(rng.normal(0, 0.3, meas.size)),
             "const": np.full(meas.size, 1.0)}
    d = AE.decide(preds, meas, pair, simple=["spread", "const"], min_items=10, n_boot=500)
    assert d["delivers"] and all(v["range_above_zero"] and v["share_pairs_better"] >= 0.75 for v in d["vs"].values())
    preds["spread"] = meas * np.exp(rng.normal(0, 0.02, meas.size))   # a better simple method -> not delivered
    d = AE.decide(preds, meas, pair, simple=["spread", "const"], min_items=10, n_boot=500)
    assert not d["delivers"]


def test_qualifying_pairs_only():
    pair = np.r_[np.zeros(30), np.ones(5)].astype(int)
    meas = np.ones(35)
    d = AE.decide({"model": meas, "spread": 2 * meas}, meas, pair, simple=["spread"], min_items=10, n_boot=100)
    assert d["n_pairs"] == 1 and d["delivers"] is None and not d["testable"]


def test_strata_split():
    rho = np.array([0.6, 0.9, 1.0, 2.0]); vza = np.array([5, 25, 45, 60.0])
    s = AE.strata(rho, vza)
    assert s["weak"].tolist() == [True, True, False, False] and s["strong"].tolist() == [False, False, True, True]
    assert s["vza<20"].tolist() == [True, False, False, False] and s["vza>=40"].tolist() == [False, False, True, True]
