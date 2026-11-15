#!/usr/bin/env python
"""One-time test of ONE frozen checkpoint on the held-out episode (Central Kalimantan, September 2015; night and day packs).

  python scripts/closed_test.py --ckpt <candidate>.pt --ckpt-sha256 <sha256> --const-json <train_const>.json \\
      --pack night=<dir> day=<dir> --out <workdir> [--execute] [--git-sha <commit of this code>]

There is no tune-on-it affordance. Everything is fixed in committed files before the test:
  configs/closed_test_amendment.json   how the four lines are read (status APPROVED, thresholds in `reader_constants`)
  configs/closed_test_candidates.json  the ONE candidate (checkpoint sha256) and the input-blind constant file (sha256)
  configs/closed_test_pack_listing.json  file names and sizes of the two packs
Without --execute the script runs every check that needs no scoring and prints the plan (no lock, nothing written).
With --execute it (1) checks the checkpoint, constant file and packs against those files, (2) takes the lock `<out>/LOCK.json`
(created exclusively: a second run into the same --out is refused, whatever the checkpoint), (3) scores the night and the day pack with
scripts/score_run.py (set val2all) and (4) reads the score files with scripts/closed_test_reader.py (night decides; day is exploratory).
The record `<out>/closed_test_record.json` and `<out>/STATUS.json` are written on success, crash and interrupt. A retry is possible only after
a failure BEFORE any score file exists (--retry-reason), for the SAME checkpoint; it writes its own record and never touches the first.
The lock is a local file: it stops an accidental second run, it cannot stop a person who deletes it. The record carries the hashes of
the scorer, the reader, the amendment and the packs so that any change is visible."""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def sha256f(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _json(path):
    return json.load(open(path))


def check_inputs(a, gate):
    """All pre-lock checks that need no scoring. -> (errors, context)."""
    errors = []
    amend = _json(os.path.join(gate, "closed_test_amendment.json"))
    cands = _json(os.path.join(gate, "closed_test_candidates.json"))
    listing = _json(os.path.join(gate, "closed_test_pack_listing.json"))
    if not (amend.get("status") == "APPROVED" and amend.get("approved_by") and amend.get("approved_at")):
        errors.append("configs/closed_test_amendment.json is not APPROVED (status, approved_by, approved_at)")
    cand = (cands.get("episode") or {}).get("candidate") or {}
    need = ("name", "file", "sha256", "const_file", "chosen_by", "chosen_at")
    if not all(cand.get(k) for k in need) or not isinstance(cand.get("why_this_candidate"), dict) \
            or not all(str(cand["why_this_candidate"].get(k, "")).strip() for k in ("pre_stated_lines_met_on_SELECT", "seeds", "known_weaknesses")):
        errors.append("no complete frozen candidate in configs/closed_test_candidates.json (fields and why_this_candidate)")
    else:
        if cand["sha256"] != a.ckpt_sha256:
            errors.append("--ckpt-sha256 differs from the frozen candidate")
        if not any(c["sha256"] == cand["sha256"] and c["file"] == cand["file"] for c in cands.get("frozen_candidates_available", [])):
            errors.append("the candidate is not in the frozen list")
    if not os.path.isfile(a.ckpt):
        errors.append(f"checkpoint {a.ckpt} not found")
    elif cand.get("sha256") and sha256f(a.ckpt) != cand["sha256"]:
        errors.append("checkpoint file sha256 differs from the frozen candidate")
    const_info = (cands.get("const_files") or {}).get(cand.get("const_file", ""))
    if not const_info:
        errors.append("the candidate's const_file is not listed in const_files")
    elif not os.path.isfile(a.const_json) or sha256f(a.const_json) != const_info["sha256"]:
        errors.append("--const-json is missing or its sha256 differs from the committed one")
    packs = dict(kv.split("=", 1) for kv in a.pack)
    if set(packs) != {"night", "day"}:
        errors.append("--pack needs night=<dir> and day=<dir>")
    for k, d in packs.items():
        pin = (listing.get("packs") or {}).get(k)
        have = sorted((os.path.relpath(os.path.join(dp, f), d), os.path.getsize(os.path.join(dp, f))) for dp, _, fn in os.walk(d) for f in fn) if os.path.isdir(d) else []
        want = sorted((o["name"], o["size"]) for o in (pin or {}).get("objects", []))
        if not pin or have != want:
            errors.append(f"pack {k}: file names/sizes differ from the pinned listing")
    from nightrad.raw.nf2_split import COMMITTED_RETIRED_SHA256, COMMITTED_SPLIT_SHA256, file_sha256
    if file_sha256(os.path.join(ROOT, "configs", "nf2_split.json")) != COMMITTED_SPLIT_SHA256:
        errors.append("configs/nf2_split.json is not the pinned split")
    if file_sha256(os.path.join(ROOT, "configs", "report_retired.json")) != COMMITTED_RETIRED_SHA256:
        errors.append("configs/report_retired.json is not the pinned retire list")
    return errors, {"cand": cand, "packs": packs, "amend_path": os.path.join(gate, "closed_test_amendment.json"), "const_info": const_info}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--ckpt-sha256", required=True, help="typed again: must equal the frozen candidate")
    p.add_argument("--const-json", required=True)
    p.add_argument("--pack", nargs=2, required=True, metavar="NAME=DIR")
    p.add_argument("--out", required=True)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--git-sha", default=None)
    p.add_argument("--retry-reason", default=None)
    p.add_argument("--gate-dir", default=os.path.join(ROOT, "configs"), help="where the three gate files live (tests only)")
    a = p.parse_args(argv)

    errors, ctx = check_inputs(a, a.gate_dir)
    cand = ctx["cand"]
    status_f, lock_f = os.path.join(a.out, "STATUS.json"), os.path.join(a.out, "LOCK.json")
    attempt = 1
    if a.retry_reason:
        st = _json(status_f) if os.path.exists(status_f) else {}
        if st.get("status") != "failed_before_score" or st.get("candidate_sha256") != a.ckpt_sha256:
            errors.append("retry refused: STATUS is not failed_before_score for this checkpoint (a score exists, or another checkpoint was locked)")
        else:
            attempt = int(st.get("attempt", 1)) + 1
    elif os.path.exists(lock_f):
        errors.append(f"the lock {lock_f} exists: this test already ran or started")
    if a.execute and (not a.git_sha or len(a.git_sha) < 7):
        errors.append("--git-sha <commit of this code> is required with --execute")
    plan = [f"candidate {cand.get('name')} (sha256 {a.ckpt_sha256[:12]}...)", f"lock {lock_f}", f"attempt {attempt}"]
    if not a.execute:
        print("DRY RUN (no lock, nothing written). PLAN:")
        for x in plan:
            print(" -", x)
        print("PRE-LOCK CHECKS:", "ALL PASS" if not errors else "FAIL")
        for e in errors:
            print("   FAIL:", e)
        return 0 if not errors else 2
    if errors:
        print("REFUSED (nothing locked):")
        for e in errors:
            print("   -", e)
        return 2

    os.makedirs(a.out, exist_ok=True)
    attempt_dir = os.path.join(a.out, f"attempt{attempt}")
    if os.path.exists(attempt_dir):
        print("REFUSED: attempt directory exists")
        return 2
    code = {f: sha256f(os.path.join(ROOT, f)) for f in ("scripts/score_run.py", "scripts/closed_test.py", "scripts/closed_test_reader.py", "src/nightrad/raw/nf2_split.py")}
    rec = {"candidate": cand, "attempt": attempt, "git_sha": a.git_sha, "code_sha256": code, "amendment_sha256": sha256f(ctx["amend_path"]), "ckpt_sha256": a.ckpt_sha256,
           "const_sha256": ctx["const_info"]["sha256"], "retry_reason": a.retry_reason, "started": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "status": "lock_taken", "runs": {},
           "packs": {k: {os.path.relpath(os.path.join(dp, f), d): sha256f(os.path.join(dp, f)) for dp, _, fn in os.walk(d) for f in sorted(fn)} for k, d in ctx["packs"].items()}}
    if attempt == 1:
        try:
            fd = os.open(lock_f, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            print("REFUSED: could not take the lock (it exists)")
            return 2
        with os.fdopen(fd, "w") as f:
            json.dump({"candidate_sha256": a.ckpt_sha256, "started": rec["started"], "git_sha": a.git_sha}, f)
    os.makedirs(attempt_dir)
    json.dump({"status": "started", "attempt": attempt, "candidate_sha256": a.ckpt_sha256}, open(status_f, "w"))

    def stop(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, stop)
    ck_dir = os.path.join(attempt_dir, "ck", cand["name"])
    os.makedirs(ck_dir)
    os.symlink(os.path.abspath(a.ckpt), os.path.join(ck_dir, "best_val.pt"))
    scorer = os.environ.get("CLOSED_TEST_SCORER") or os.path.join(ROOT, "scripts", "score_run.py")   # the environment override is for the tests only
    try:
        for k, d in ctx["packs"].items():
            cmd = [sys.executable, scorer, "--data", d, "--train", "UNUSED", "--const-json", a.const_json, "--set", "val2all", "--runs-dir", os.path.join(attempt_dir, "ck"),
                   "--names", cand["name"], "--ckpt", "best_val.pt", "--out", os.path.join(attempt_dir, f"out_{k}"), "--git-sha", a.git_sha,
                   "--allow-unnamed-reason", "held-out episode ids are absent from the training, selection and report lists by construction"]
            t0 = time.time()
            r = subprocess.run(cmd, cwd=ROOT, env={**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")})
            rec["runs"][k] = {"returncode": r.returncode, "seconds": round(time.time() - t0, 1)}
            if r.returncode != 0:
                raise RuntimeError(f"scorer failed for {k} (returncode {r.returncode})")
        reader = os.path.join(ROOT, "scripts", "closed_test_reader.py")
        rec["reader_sha256"] = sha256f(reader)
        for k, flag in (("night", []), ("day", ["--exploratory"])):
            sf = os.path.join(attempt_dir, f"out_{k}", cand["name"], "score_val2all.json")
            r2 = subprocess.run([sys.executable, reader, sf, "--amendment", ctx["amend_path"], *flag, "--out", os.path.join(attempt_dir, f"verdict_{k}.json")], capture_output=True, text=True)
            rec["runs"][f"reader_{k}"] = {"returncode": r2.returncode}
            if r2.returncode != 0:
                rec["reader_error"] = (rec.get("reader_error", "") + f" {k}: {r2.stderr[-200:]}").strip()
            else:
                rec[f"verdict_{k}"] = _json(os.path.join(attempt_dir, f"verdict_{k}.json"))["overall"]
        rec["status"] = "completed"
    except BaseException as e:   # includes KeyboardInterrupt and SIGTERM
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["status"] = "interrupted" if isinstance(e, KeyboardInterrupt) else "failed"
    finally:
        produced = sorted(os.path.join(dp, f) for dp, _, fn in os.walk(attempt_dir) for f in fn if f.startswith(("score_", "verdict_")) and f.endswith(".json"))
        rec["outputs"] = produced
        rec["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
        done = rec["status"] == "completed"
        st = {"status": "completed" if done else ("failed_before_score" if not produced else "failed_after_some_score"), "detail": rec["status"], "attempt": attempt, "candidate_sha256": a.ckpt_sha256}
        json.dump(rec, open(os.path.join(attempt_dir, "closed_test_record.json"), "w"), indent=1, default=str)
        json.dump(st, open(status_f, "w"))
        print("STATUS", st["status"], "| record", os.path.join(attempt_dir, "closed_test_record.json"))
    return 0 if rec["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
