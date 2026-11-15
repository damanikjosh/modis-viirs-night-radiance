import os, subprocess, sys
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "paper"))


@pytest.fixture(scope="session")
def gen_out(tmp_path_factory):
    """Run scripts/paper/gen_numbers.py once on the stored results (manifest checked by the script) and return the output directory."""
    out = tmp_path_factory.mktemp("paper_numbers")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "paper", "gen_numbers.py"), "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    return out
