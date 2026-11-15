"""Download the six granules of every pair of a pair table from NASA (LAADS DAAC for L1B and geolocation, LP DAAC for fire products).

Login (free Earthdata account, https://urs.earthdata.nasa.gov; approve the LAADS DAAC and LP DAAC applications):
  * environment variable NR_EARTHDATA_TOKEN (a user token, sent as a bearer header), or
  * a ~/.netrc entry `machine urs.earthdata.nasa.gov login <user> password <password>` (chmod 600).
Layout: <root>/<pair_id>/<PRODUCT>/<granule file>, root = $NR_DATA_ROOT/raw by default. Resumable; existing complete files are skipped.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

from .pairs import product_files


class LoginError(RuntimeError):
    pass


def data_root() -> Path:
    """Root of every downloaded and derived file: the environment variable NR_DATA_ROOT (default ./data)."""
    return Path(os.environ.get("NR_DATA_ROOT", "data"))


def login() -> requests.Session:
    """Authenticated requests session (NR_EARTHDATA_TOKEN, else ~/.netrc through earthaccess)."""
    tok = os.environ.get("NR_EARTHDATA_TOKEN")
    if tok:
        s = requests.Session()
        s.headers["Authorization"] = f"Bearer {tok}"
        return s
    import earthaccess
    try:
        auth = earthaccess.login(strategy="netrc")
    except Exception as e:  # earthaccess raises generic errors when no credentials exist
        raise LoginError(_msg(e)) from e
    if not getattr(auth, "authenticated", False):
        raise LoginError(_msg("not authenticated"))
    return earthaccess.get_requests_https_session()


def _msg(e):
    return (f"Earthdata login failed ({e}). Set NR_EARTHDATA_TOKEN, or create ~/.netrc with\n"
            "  machine urs.earthdata.nasa.gov login <user> password <pass>\nand chmod 600 ~/.netrc "
            "(register free at https://urs.earthdata.nasa.gov, approve the LP DAAC and LAADS apps).")


def fetch(session, url: str, dest: Path, chunk: int = 1 << 20) -> str:
    """Download url to dest via dest.part with Range resume; the size must equal the server's Content-Length.
    Returns 'skipped' | 'done'."""
    if dest.exists() and dest.stat().st_size > 0:
        return "skipped"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with session.get(url, headers=headers, stream=True, timeout=(30, 60), allow_redirects=True) as r:
        if r.status_code == 416:  # already complete
            part.rename(dest)
            return "done"
        r.raise_for_status()
        resumed = have and r.status_code == 206
        expect = int(r.headers["Content-Length"]) + (have if resumed else 0) if "Content-Length" in r.headers else None
        with open(part, "ab" if resumed else "wb") as f:
            for b in r.iter_content(chunk):
                f.write(b)
    if expect is not None and part.stat().st_size != expect:
        raise OSError(f"{dest.name}: got {part.stat().st_size} bytes, expected {expect}")
    part.rename(dest)
    return "done"


def pair_dir(root: str | Path, pair_id: str) -> Path:
    return Path(root) / pair_id


def fetch_retry(session, url: str, dest: Path, attempts: int = 4) -> str:
    """`fetch` with retries (connection errors, timeouts, short reads); the partial file is resumed."""
    for i in range(attempts):
        try:
            return fetch(session, url, dest)
        except (requests.RequestException, OSError):
            if i == attempts - 1:
                raise
            time.sleep(5 * (i + 1))


def download_pair(session, row, root: str | Path) -> list[Path]:
    """Fetch the six granules of one pair; returns their paths."""
    paths = []
    for prod, _gid, url in product_files(row):
        dest = pair_dir(root, row["pair_id"]) / prod / url.rsplit("/", 1)[1]
        fetch_retry(session, url, dest)
        paths.append(dest)
    return paths


def download(pairs: pd.DataFrame, root: str | Path | None = None, threads: int = 4, log=print) -> None:
    """Download every pair of the table (rows need pair_id and the url columns of `pairs.attach_granules`)."""
    root = Path(root) if root else data_root() / "raw"
    session = login()
    rows = pairs.to_dict("records")

    def one(r):
        download_pair(session, r, root)
        return r["pair_id"]

    with ThreadPoolExecutor(threads) as ex:
        for i, pid in enumerate(ex.map(one, rows), 1):
            log(f"[{i}/{len(rows)}] {pid}")
