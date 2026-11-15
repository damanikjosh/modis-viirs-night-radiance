import json, os, sys, tarfile, io, subprocess
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "design"))
import guard
from nightrad.raw import nf2_split as NS

SPLIT = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "nf2_split.json")


def test_allowed_excludes_report_and_retired():
    a = guard.allowed_ids(SPLIT)
    t = json.load(open(SPLIT))["tiles"]
    assert set(a) == set(t["train"]) | set(t["select"])
    assert not (set(a) & set(t["report"])) and not (set(a) & NS.retired_ids(NS.DEFAULT_RETIRED))


def test_report_side_refused():
    with pytest.raises(SystemExit):
        guard.allowed_ids(SPLIT, sides=("report",))
    with pytest.raises(SystemExit):
        guard.allowed_ids(SPLIT, sides=("train", "report"))


def test_report_id_refused_by_check_ids():
    rep = json.load(open(SPLIT))["tiles"]["report"][0]
    with pytest.raises(SystemExit):
        guard.check_ids([rep], guard.allowed_ids(SPLIT))


def test_extractor_never_reads_a_report_member(tmp_path):
    t = json.load(open(SPLIT))["tiles"]; rep, tr = t["report"][0], t["train"][0]
    tar = tmp_path / "x.tar"
    with tarfile.open(tar, "w") as tf:
        for n in (rep, tr):
            b = b"data-" + n.encode(); ti = tarfile.TarInfo(n + ".npz"); ti.size = len(b); tf.addfile(ti, io.BytesIO(b))
    out = tmp_path / "o"
    r = subprocess.run([sys.executable, os.path.join(os.path.dirname(guard.__file__), "extract_train.py"), str(out), str(tar)], capture_output=True, text=True,
                       cwd=os.path.join(os.path.dirname(__file__), "..", ".."))
    assert r.returncode == 0, r.stderr
    assert sorted(os.listdir(out)) == sorted([tr + ".npz", "_extract_report.json"])
