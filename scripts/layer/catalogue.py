#!/usr/bin/env python
"""Layer step 1: granule catalogue. Aqua MYD021KM + MYD03 night granules over a bounding box and time window from NASA CMR (public metadata, no login), each paired with its
MYD14 file (read later only for its cloud / water / quality classes; fire detections are not an input).

usage: catalogue.py --bbox W,S,E,N --start YYYY-MM-DD --end YYYY-MM-DD [--ids A2006253.1815,...] [--out catalogue.jsonl]
One JSON line per granule: id (A{yyyyddd}.{hhmm}), t (UTC iso start time), l1b / geo / fm = {name, https, s3, size_mb} (fm = MYD14, null when CMR has none)."""
import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, timedelta

CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"


def _query(short, bbox, t0, t1, night=True, version="6.1"):
    out, page = [], 1
    while True:
        p = {"short_name": short, "version": version, "bounding_box": bbox, "temporal": f"{t0}T00:00:00Z,{t1}T00:00:00Z", "page_size": 2000, "page_num": page}
        if night:
            p["day_night_flag"] = "night"
        url = CMR + "?" + urllib.parse.urlencode(p)
        for k in range(5):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    d = json.load(r)["feed"]["entry"]
                break
            except Exception:
                time.sleep(2 * (k + 1))
        else:
            raise RuntimeError("CMR request failed: " + url)
        out += d
        if len(d) < 2000:
            return out
        page += 1


def _record(e):
    https = next(link["href"] for link in e["links"] if link["href"].startswith("https://data.") and link["href"].endswith(".hdf"))
    s3 = next((link["href"] for link in e["links"] if link["href"].startswith("s3://")), None)
    return {"name": https.rsplit("/", 1)[1], "https": https, "s3": s3, "size_mb": float(e.get("granule_size", 0) or 0), "t": e["time_start"]}


def _by_id(entries):
    out = {}
    for e in entries:
        r = _record(e)
        out[".".join(r["name"].split(".")[1:3])] = r
    return out


def catalogue(bbox, start, end, step_days=31, ids=None):
    d0, d1, res = date.fromisoformat(start), date.fromisoformat(end), []
    while d0 < d1:
        n = min(d0 + timedelta(days=step_days), d1)
        a = _by_id(_query("MYD021KM", bbox, d0.isoformat(), n.isoformat()))
        g = _by_id(_query("MYD03", bbox, d0.isoformat(), n.isoformat()))
        m = _by_id(_query("MYD14", bbox, d0.isoformat(), n.isoformat(), version="061"))
        for k in sorted(a):
            if ids is None or k in ids:
                res.append({"id": k, "t": a[k]["t"], "l1b": a[k], "geo": g.get(k), "fm": m.get(k)})
        d0 = n
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bbox", required=True, help="W,S,E,N in degrees")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--ids", default=None, help="comma-separated granule ids to keep (A{yyyyddd}.{hhmm})")
    ap.add_argument("--out", default="-")
    a = ap.parse_args()
    f = sys.stdout if a.out == "-" else open(a.out, "w")
    for r in catalogue(a.bbox, a.start, a.end, ids=set(a.ids.split(",")) if a.ids else None):
        f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
