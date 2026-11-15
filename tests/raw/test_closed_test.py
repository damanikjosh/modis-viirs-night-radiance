"""closed_test.py: every refusal and the lock / record behaviour, with a stub scorer and synthetic files (never the real packs)."""
import hashlib
import importlib.util
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location("closed_test", os.path.join(ROOT, "scripts", "closed_test.py"))
CT = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CT)

STUB = '''
import argparse, json, os, signal, sys, time
p = argparse.ArgumentParser()
for k in ("data", "out", "names", "train", "const-json", "set", "runs-dir", "ckpt", "git-sha", "allow-unnamed-reason"):
    p.add_argument("--" + k)
a = p.parse_args(); mode = os.environ.get("STUB_MODE", "ok")
if mode == "crash":
    sys.exit(3)
if mode == "crash_day" and a.data.endswith("pack_day"):
    sys.exit(4)
if mode == "sigint":
    os.kill(os.getppid(), signal.SIGTERM); time.sleep(5)
os.makedirs(os.path.join(a.out, a.names), exist_ok=True)
json.dump({"run": a.names, "data": a.data, "n_tiles": 3}, open(os.path.join(a.out, a.names, "score_val2all.json"), "w"))
'''


@pytest.fixture
def env(tmp_path, monkeypatch):
    gate = tmp_path / "gate"
    gate.mkdir()
    ck = tmp_path / "cand__best_val.pt"
    ck.write_bytes(b"checkpoint-bytes")
    sha = hashlib.sha256(b"checkpoint-bytes").hexdigest()
    const = tmp_path / "const.json"
    const.write_text(json.dumps({"const": -0.36}))
    packs = {}
    listing = {"packs": {}}
    for k in ("night", "day"):
        d = tmp_path / f"pack_{k}"
        d.mkdir()
        objs = []
        for i in range(3):
            b = f"{k}-{i}".encode() * 10
            (d / f"f{i}.npy").write_bytes(b)
            objs.append({"name": f"f{i}.npy", "size": len(b)})
        packs[k] = d
        listing["packs"][k] = {"n_objects": 3, "objects": objs}
    json.dump(listing, open(gate / "closed_test_pack_listing.json", "w"))
    reader_constants = json.load(open(os.path.join(ROOT, "configs", "closed_test_amendment.json")))["reader_constants"]
    json.dump({"status": "APPROVED", "approved_by": "lead", "approved_at": "2026-01-01 09:05", "reader_constants": reader_constants}, open(gate / "closed_test_amendment.json", "w"))
    cand = {"name": "cand", "file": "cand__best_val.pt", "sha256": sha, "const_file": "c", "chosen_by": "lead", "chosen_at": "09:06",
            "why_this_candidate": {"pre_stated_lines_met_on_SELECT": "4 of 5", "seeds": "2 of 4", "known_weaknesses": "line 1"}}
    json.dump({"frozen_candidates_available": [{"name": "cand", "file": "cand__best_val.pt", "sha256": sha}],
               "const_files": {"c": {"sha256": hashlib.sha256(const.read_bytes()).hexdigest()}}, "episode": {"candidate": cand}}, open(gate / "closed_test_candidates.json", "w"))
    stub = tmp_path / "stub.py"
    stub.write_text(STUB)
    monkeypatch.setenv("CLOSED_TEST_SCORER", str(stub))

    class E:
        pass
    e = E()
    e.gate, e.sha, e.tmp, e.ck, e.const, e.packs, e.out = gate, sha, tmp_path, ck, const, packs, tmp_path / "out"

    def run(*extra, sha_arg=None, execute=True, git_sha="abcdef1", out=None):
        args = ["--ckpt", str(e.ck), "--ckpt-sha256", sha_arg or sha, "--const-json", str(e.const), "--pack", f"night={packs['night']}", f"day={packs['day']}",
                "--out", str(out or e.out), "--gate-dir", str(gate), *extra]
        if git_sha:
            args += ["--git-sha", git_sha]
        return CT.main(args + (["--execute"] if execute else []))
    e.run = run
    return e


def _amend(env, **kw):
    f = env.gate / "closed_test_amendment.json"
    a = json.load(open(f))
    a.update(kw)
    json.dump(a, open(f, "w"))


def test_dry_run_takes_no_lock_and_writes_nothing(env, capsys):
    assert env.run(execute=False) == 0
    assert "ALL PASS" in capsys.readouterr().out and not env.out.exists()


def test_unapproved_amendment_is_refused(env):
    _amend(env, status="PROPOSED")
    assert env.run() == 2 and not env.out.exists()


def test_wrong_typed_sha_checkpoint_const_and_pack_are_refused_before_the_lock(env):
    assert env.run(sha_arg="0" * 64) == 2
    env.ck.write_bytes(b"other")
    assert env.run() == 2
    env.ck.write_bytes(b"checkpoint-bytes")
    env.const.write_text("{}")
    assert env.run() == 2
    env.const.write_text(json.dumps({"const": -0.36}))
    (env.packs["night"] / "f0.npy").write_bytes(b"x")
    assert env.run() == 2
    assert not env.out.exists()


def test_candidate_without_why_this_candidate_is_refused(env):
    f = env.gate / "closed_test_candidates.json"
    c = json.load(open(f))
    c["episode"]["candidate"]["why_this_candidate"]["known_weaknesses"] = ""
    json.dump(c, open(f, "w"))
    assert env.run() == 2


def test_git_sha_is_required_to_execute(env):
    assert env.run(git_sha=None) == 2 and not env.out.exists()


def test_happy_path_takes_one_lock_and_writes_record_status_and_verdicts(env):
    assert env.run() == 0
    assert os.path.exists(env.out / "LOCK.json")
    rec = json.load(open(env.out / "attempt1" / "closed_test_record.json"))
    assert rec["status"] == "completed" and rec["ckpt_sha256"] == env.sha and rec["verdict_day"] == "EXPLORATORY_NO_VERDICT" and "verdict_night" in rec
    assert set(rec["packs"]) == {"night", "day"} and len(rec["packs"]["night"]) == 3
    assert {"scripts/score_run.py", "scripts/closed_test_reader.py"} <= set(rec["code_sha256"])
    assert json.load(open(env.out / "STATUS.json"))["status"] == "completed"
    assert len(rec["outputs"]) == 4


def test_second_run_is_refused_whatever_the_name_or_checkpoint(env):
    assert env.run() == 0
    assert env.run() == 2
    assert env.run(out=env.out) == 2
    assert env.run(execute=False) == 2


def test_scorer_crash_keeps_the_lock_and_a_retry_needs_a_reason_and_the_same_checkpoint(env, monkeypatch):
    monkeypatch.setenv("STUB_MODE", "crash")
    assert env.run() == 1
    assert json.load(open(env.out / "STATUS.json"))["status"] == "failed_before_score"
    assert env.run() == 2                                  # no reason: refused
    monkeypatch.setenv("STUB_MODE", "ok")
    assert env.run("--retry-reason", "scorer crashed before any score") == 0
    rec1 = json.load(open(env.out / "attempt1" / "closed_test_record.json"))
    rec2 = json.load(open(env.out / "attempt2" / "closed_test_record.json"))
    assert rec1["status"] == "failed" and rec2["status"] == "completed" and rec2["retry_reason"]
    assert env.run("--retry-reason", "again") == 2         # a completed test cannot be retried


def test_failure_after_one_pack_was_scored_forbids_a_retry(env, monkeypatch):
    monkeypatch.setenv("STUB_MODE", "crash_day")
    assert env.run() == 1
    st = json.load(open(env.out / "STATUS.json"))
    assert st["status"] == "failed_after_some_score"
    monkeypatch.setenv("STUB_MODE", "ok")
    assert env.run("--retry-reason", "try again") == 2


def test_termination_writes_record_and_status(env, monkeypatch):
    monkeypatch.setenv("STUB_MODE", "sigint")  # the stub terminates its parent
    assert env.run() == 1
    rec = json.load(open(env.out / "attempt1" / "closed_test_record.json"))
    assert rec["status"] == "interrupted"
    assert json.load(open(env.out / "STATUS.json"))["status"] == "failed_before_score"
