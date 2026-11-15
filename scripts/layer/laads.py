"""Download from NASA LAADS DAAC with an Earthdata login: bearer token from the environment variable NR_EARTHDATA_TOKEN, else the machine entry for
urs.earthdata.nasa.gov in ~/.netrc. Credentials are never printed or written anywhere. Downloads go to <name>.part and are renamed only when complete."""
import netrc
import os
import time
from pathlib import Path

import requests

TOKEN_ENV = "NR_EARTHDATA_TOKEN"
RETRY_CODES = (429, 500, 502, 503, 504)


def session() -> requests.Session:
    s = requests.Session()
    tok = os.environ.get(TOKEN_ENV, "").strip()
    if tok:
        s.headers["Authorization"] = "Bearer " + tok  # requests drops this header when a redirect leaves the host
        return s
    try:
        auth = netrc.netrc().authenticators("urs.earthdata.nasa.gov")
    except (FileNotFoundError, netrc.NetrcParseError):
        auth = None
    if auth is None:
        raise SystemExit(f"no Earthdata login: set {TOKEN_ENV} (an Earthdata user token) or add 'machine urs.earthdata.nasa.gov login <user> password <pass>' to ~/.netrc (chmod 600)")
    return s  # requests reads ~/.netrc itself when the download is redirected to urs.earthdata.nasa.gov


def fetch(s: requests.Session, url: str, dst: Path, expected_mb: float = 0.0, tries: int = 6):
    """-> (True, status) when dst exists and is complete (at least 97 % of the catalogue size; CMR sizes are in MiB), else (False, last status)."""
    dst = Path(dst)
    if dst.exists() and dst.stat().st_size > 0:
        return True, "cached"
    part = Path(str(dst) + ".part")
    dst.parent.mkdir(parents=True, exist_ok=True)
    delay, code = 5, "error"
    for _ in range(tries):
        try:
            with s.get(url, stream=True, timeout=(30, 120), allow_redirects=True) as r:
                code = r.status_code
                if code == 200:
                    with open(part, "wb") as f:
                        for b in r.iter_content(1 << 20):
                            f.write(b)
                    if part.stat().st_size > 0 and (not expected_mb or part.stat().st_size >= 0.97 * expected_mb * 1048576):
                        part.rename(dst)
                        return True, 200
                    code = "truncated"
                elif code not in RETRY_CODES:
                    return False, code
        except requests.RequestException as e:
            code = type(e).__name__
        time.sleep(delay)
        delay = min(delay * 2, 120)
    return False, code
