import numpy as np
from nightrad.raw import nf2_split as S


def test_blocks_link_within_100km_and_are_transitive():
    lat = np.array([0.0, 0.0, 0.0, 0.0, 10.0]); lon = np.array([0.0, 0.8, 1.6, 5.0, 0.0])  # 0.8 deg ~ 89 km: 0-1-2 chain (0-2 is 178 km), 3 and 4 alone
    b = S.blocks(lat, lon, 100.0)
    assert b[0] == b[1] == b[2] and len({b[0], b[3], b[4]}) == 3
    assert S.blocks(lat, lon, 50.0).tolist() == [0, 1, 2, 3, 4]


def test_antimeridian_and_poles_are_handled_by_3d_distance():
    b = S.blocks([0.0, 0.0], [179.6, -179.6], 100.0)
    assert b[0] == b[1]


def test_assign_never_splits_a_block_is_deterministic_and_balances_fire():
    rng = np.random.default_rng(1)
    n = 400; lat = rng.uniform(-30, 30, n); lon = rng.uniform(100, 150, n)
    block = S.blocks(lat, lon, 100.0); region = np.where(lat > 0, "N", "S"); fire = rng.integers(0, 5, n) * (rng.random(n) < 0.3)
    s1 = S.assign(region, block, fire, np.arange(n), seed=3); s2 = S.assign(region, block, fire, np.arange(n), seed=3)
    assert s1 == s2 and set(s1) == set(block.tolist())
    side = np.array([s1[b] for b in block])
    assert min_dist_cross(lat, lon, side) >= 100.0
    for r in "NS":
        f = {k: fire[(region == r) & (side == k)].sum() for k in ("select", "report")}
        assert abs(f["select"] - f["report"]) <= max(1, 0.35 * fire[region == r].sum())


def min_dist_cross(lat, lon, side):
    a, b = side == "select", side == "report"
    return S.min_dist_km(lat[a], lon[a], lat[b], lon[b])


def test_side_indices_cover_each_tile_once_and_refuse_unlisted(tmp_path):
    import json
    import pytest
    f = tmp_path / "s.json"; json.dump({"tiles": {"train": ["a", "b"], "select": ["c", "d"], "report": ["e"]}}, open(f, "w"))
    assert S.side_indices(["c", "d", "e"], f, "select") == [0, 1] and S.side_indices(["c", "d", "e"], f, "report") == [2]
    with pytest.raises(AssertionError):
        S.side_indices(["c", "zzz"], f, "select")  # pack tile not in any list
    json.dump({"tiles": {"train": ["a"], "select": ["a"], "report": []}}, open(f, "w"))
    with pytest.raises(AssertionError):
        S.side_indices(["a"], f, "select")  # tile in two lists


def test_sample_weights_and_epoch_checkpoints_in_training(tmp_path):
    """--sample-weights drive the draws (weight 0 tiles are never drawn) and --ckpt-every-epoch writes one model file per epoch."""
    import os
    from argparse import Namespace
    from nightrad.raw.core import RunConfig
    from nightrad.raw.net import NetConfig
    from nightrad.raw.synth import SyntheticTiles
    from nightrad.raw.train import train
    n = 16; w = np.ones(n); w[:8] = 0.0; np.save(tmp_path / "w.npy", w)
    seen = []

    class Rec(SyntheticTiles):
        def __getitem__(self, i):
            seen.append(i); return super().__getitem__(i)
    a = Namespace(out=str(tmp_path / "o"), epochs=2, bs=4, lr=2e-3, wd=1e-4, warmup=2, amp="off", channels_last=False, compile=False, workers=0, seed=1, resume=False, deterministic=False,
                  clip=1.0, sample_weights=str(tmp_path / "w.npy"), ckpt_every_epoch=True, no_legacy_val=True)
    train(a, Rec(n, 0, 48, 48, cache=False), SyntheticTiles(8, 10**6, 48, 48, cache=True), RunConfig(net=NetConfig(width=8, latent=4), daynight="both"), log=lambda *_: None)
    assert seen and min(seen) >= 8, "a zero-weight tile was drawn"
    assert os.path.exists(tmp_path / "o" / "ep001.pt") and os.path.exists(tmp_path / "o" / "ep002.pt")


def _pack(d, names):
    import json
    d.mkdir(exist_ok=True); json.dump(names, open(d / "names.json", "w")); json.dump({"n": len(names), "keys": []}, open(d / "meta.json", "w"))


def test_scorer_guard_refuses_report_tiles_and_unnamed_packs(tmp_path, monkeypatch):
    import json
    import pytest
    split = tmp_path / "s.json"; json.dump({"tiles": {"train": ["a"], "select": ["b", "c"], "report": ["d", "e"]}}, open(split, "w"))
    monkeypatch.setattr(S, "COMMITTED_SPLIT_SHA256", S.file_sha256(split))   # unit test with a toy split file
    ok = tmp_path / "ok"; _pack(ok, ["b", "c"])
    assert S.check_scoring_set(str(ok), str(split), "select")["checked"]
    bad = tmp_path / "bad"; _pack(bad, ["b", "d"])   # one REPORT tile hidden among SELECT tiles
    with pytest.raises(SystemExit):
        S.check_scoring_set(str(bad), str(split), "select")
    with pytest.raises(SystemExit):
        S.check_scoring_set(str(bad), str(split), "report", final=False)      # REPORT set without --final is refused too
    assert S.check_scoring_set(str(bad), str(split), "report", final=True)["checked"]   # stage-5 final scoring of the report set is the only allowed use
    nonames = tmp_path / "nn"; nonames.mkdir()
    with pytest.raises(SystemExit):
        S.check_scoring_set(str(nonames), str(split), "select")
    assert not S.check_scoring_set(str(nonames), str(split), "select", allow_unnamed_reason="old v3 validation, ids not in nf2")["checked"]
    x = json.load(open("configs/xdata_list.json"))["tiles"]; real = json.load(open("configs/nf2_split.json"))["tiles"]
    assert len(set(x) & set(real["report"])) == 88          # the invalid cross-data list is exactly what the guard must stop


def test_scorer_guard_checks_only_the_scored_subset(tmp_path, monkeypatch):
    import json
    import pytest
    split = tmp_path / "s.json"; json.dump({"tiles": {"train": ["a"], "select": ["b", "c"], "report": ["d", "e"]}}, open(split, "w"))
    monkeypatch.setattr(S, "COMMITTED_SPLIT_SHA256", S.file_sha256(split))
    pk = tmp_path / "pk"; _pack(pk, ["b", "c", "d", "e"])   # the val2 pack holds SELECT and REPORT tiles
    assert S.check_scoring_set(str(pk), str(split), "select", indices=[0, 1])["tiles_checked"] == 2          # scoring only the SELECT indices is allowed
    with pytest.raises(SystemExit):
        S.check_scoring_set(str(pk), str(split), "select", indices=[0, 2])                           # one REPORT index among the scored tiles is refused
    with pytest.raises(SystemExit):
        S.check_scoring_set(str(pk), str(split), "select")                                           # no subset given: the whole pack counts
