"""Worker bookkeeping (version rules, resume, failure records), checkpoint check, download credentials. No network, no GPU."""
import importlib.util
import json
import pathlib

import pytest

import laads
from nightrad.inference import version as V

spec = importlib.util.spec_from_file_location("worker", pathlib.Path(__file__).parents[2] / "scripts/layer/worker.py")
W = importlib.util.module_from_spec(spec); spec.loader.exec_module(W)


def test_version_ordering_and_default():
    assert V.PIPELINE_VERSION == "v1.2" and V.vkey("v1.2") > V.vkey("v1.1") > V.vkey(None) == (1, 0) and V.vkey("v1.10") > V.vkey("v1.2") and V.vkey("garbage") == (0, 0)


def test_checkpoint_hash_is_enforced(tmp_path):
    bad = tmp_path / "x.pt"; bad.write_bytes(b"not the checkpoint")
    with pytest.raises(SystemExit, match="sha256 mismatch"):
        V.verify_checkpoint(bad)
    if V.DEFAULT_CKPT.exists():
        assert V.verify_checkpoint(V.DEFAULT_CKPT) == V.CKPT_SHA256


def test_ok_record_resume_and_never_overwrites_newer(tmp_path):
    assert not W.is_done(tmp_path, "G")
    W.put_record(tmp_path, {"id": "G", "status": "ok", "pipeline_version": "v1.2"})
    assert W.is_done(tmp_path, "G")
    W.put_record(tmp_path, {"id": "G", "status": "failed", "pipeline_version": "v1.2", "error": "x"})        # a failed re-run never replaces an ok record of the same version
    assert json.loads((tmp_path / "_done" / "G.json").read_text())["status"] == "ok" and not (tmp_path / "_failed" / "G.json").exists()
    (tmp_path / "_done" / "H.json").parent.mkdir(exist_ok=True)
    (tmp_path / "_done" / "H.json").write_text(json.dumps({"id": "H", "status": "ok", "pipeline_version": "v1.1"}))
    assert not W.is_done(tmp_path, "H")                                                                           # an older ok record does not count: the granule is redone
    (tmp_path / "_done" / "N.json").write_text(json.dumps({"id": "N", "status": "ok", "pipeline_version": "v9.0", "marker": 1}))
    W.put_record(tmp_path, {"id": "N", "status": "ok", "pipeline_version": "v1.2"})
    assert json.loads((tmp_path / "_done" / "N.json").read_text())["marker"] == 1


def test_failed_record_counts_attempts_and_success_clears_it(tmp_path):
    for n in (1, 2, 3):
        W.put_record(tmp_path, {"id": "F", "status": "failed", "error": "e"})
        assert json.loads((tmp_path / "_failed" / "F.json").read_text())["attempt"] == n
    W.put_record(tmp_path, {"id": "F", "status": "ok", "pipeline_version": "v1.2"})
    assert not (tmp_path / "_failed" / "F.json").exists() and W.is_done(tmp_path, "F")


def test_laads_token_and_netrc(monkeypatch, tmp_path):
    monkeypatch.setenv("NR_EARTHDATA_TOKEN", "abc")
    assert laads.session().headers["Authorization"] == "Bearer abc"
    monkeypatch.delenv("NR_EARTHDATA_TOKEN"); monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(SystemExit, match="no Earthdata login"):
        laads.session()
    (tmp_path / ".netrc").write_text("machine urs.earthdata.nasa.gov login u password p\n"); (tmp_path / ".netrc").chmod(0o600)
    assert "Authorization" not in laads.session().headers


def test_fetch_uses_cache_and_rejects_truncated(tmp_path, monkeypatch):
    f = tmp_path / "a.hdf"; f.write_bytes(b"x" * 10)
    assert laads.fetch(None, "u", f) == (True, "cached")

    class R:
        status_code = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def iter_content(self, n): yield b"y" * 100
    class S:
        def get(self, *a, **k): return R()
    monkeypatch.setattr(laads.time, "sleep", lambda s: None)
    ok, code = laads.fetch(S(), "u", tmp_path / "b.hdf", expected_mb=1.0, tries=2)       # 100 bytes vs 1 MiB expected: truncated, never renamed
    assert not ok and code == "truncated" and not (tmp_path / "b.hdf").exists()
    ok, code = laads.fetch(S(), "u", tmp_path / "c.hdf", expected_mb=0.0, tries=1)
    assert ok and (tmp_path / "c.hdf").read_bytes() == b"y" * 100
