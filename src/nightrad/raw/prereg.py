"""Pre-registration and the test-split guard. Tuning uses train + val only; a split whose name starts with 'test' cannot be opened
without allow_test=True, and every such opening is appended to a log with time, git SHA, checkpoint hash and the hash of the frozen
config (which must exist and match)."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time


def config_hash(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def freeze(cfg: dict, path: str) -> str:
    """Write the config that will be used for ALL test evaluation (commit it before any test tile is read). Returns its hash."""
    h = config_hash(cfg)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rec = {"hash": h, "config": cfg}
    if isinstance(cfg.get("run"), dict):
        rec["run_hash"] = config_hash(cfg["run"])  # evaluation compares the RunConfig of the checkpoint against this
    json.dump(rec, open(path, "w"), indent=1, sort_keys=True, default=str)
    return h


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def file_hash(path: str | None) -> str:
    if not path or not os.path.exists(path):
        return "none"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()[:16]


def is_test_split(name: str) -> bool:
    """A split is TEST if its folder name (also after resolving symlinks), or the split field the packer recorded inside the data (pack_report.json 'split'),
    says so. (Review fix: a label or a symlinked folder name alone must not decide.)"""
    for n in {os.path.normpath(name), os.path.realpath(name)}:
        if os.path.basename(n).lower().startswith("test"):
            return True
    for d in (name, os.path.realpath(name)):
        rep = os.path.join(d, "pack_report.json")
        if os.path.exists(rep):
            try:
                if str(json.load(open(rep)).get("split", "")).lower().startswith("test"):
                    return True
            except (OSError, ValueError):
                return True  # unreadable report: refuse rather than guess
    return False


def open_split(name: str, *, allow_test: bool = False, frozen_path: str | None = None, cfg: dict | None = None,
               log_path: str = "test_access.log", ckpt: str | None = None) -> str:
    """Returns `name` if the split may be opened. Test splits need allow_test, a frozen config file and (if cfg is given) a matching hash."""
    if not is_test_split(name):
        return name
    if not allow_test:
        raise PermissionError(f"split '{name}' is a TEST split: pass allow_test=True (--allow-test); the use is logged")
    if not frozen_path or not os.path.exists(frozen_path):
        raise PermissionError("test access needs a frozen (pre-registered) config file: run prereg.freeze first and commit it")
    fz = json.load(open(frozen_path))
    if cfg is not None and config_hash(cfg) != fz.get("run_hash", fz["hash"]):
        raise PermissionError("config differs from the frozen pre-registered config: report the change instead of evaluating")
    with open(log_path, "a") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "split": name, "git": git_sha(), "config": fz.get("run_hash", fz["hash"]), "ckpt": file_hash(ckpt)}) + "\n")
    return name
