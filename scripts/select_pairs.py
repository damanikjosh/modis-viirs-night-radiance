"""Choose the night-fire pairs to tile from the FIRMS archive (eligibility, candidate nights, CMR pairs, quota per region).

  python scripts/select_pairs.py [--firms DIR] [--existing PAIRS.csv] [--quota 900] [--out PAIRS.csv]

Needs scripts/fetch_firms.py first. The paper's own list (1,017 pairs) is stored in src/nightrad/data/resources/pairs_paper.csv; re-running the
selection against today's archive and CMR gives a similar but not identical list. Rules: nightrad.data.nightfire.
Default output $NR_DATA_ROOT/pairs/pairs_selected.csv."""
import argparse
from pathlib import Path

from nightrad.data.download import data_root
from nightrad.data.nightfire import COUNTRY_REGION, eligibility, load_firms, region_of_detections, select_pairs
from nightrad.data.tables import read_centres, read_pairs

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--firms")
ap.add_argument("--existing", help="pair table already chosen: its pairs are not repeated and count towards the quota")
ap.add_argument("--quota", type=float, default=900.0, help="expected fire-level MODIS pixels per region outside South-East Asia")
ap.add_argument("--rate", type=float, default=0.7, help="expected share of eligible Aqua detections that become fire-level pixels")
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--out")
a = ap.parse_args()
det = load_firms(Path(a.firms) if a.firms else data_root() / "firms", sorted(COUNTRY_REGION))
det["region"] = region_of_detections(det)
det = eligibility(det, read_centres())
print(det.groupby(["sensor", "why"]).size().to_string())
sel = select_pairs(det, read_pairs(a.existing) if a.existing else None, a.quota, a.rate, a.threads)
out = Path(a.out) if a.out else data_root() / "pairs" / "pairs_selected.csv"
out.parent.mkdir(parents=True, exist_ok=True)
sel.to_csv(out, index=False)
print(f"wrote {out}: {len(sel)} pairs")
print(sel.groupby("region").agg(pairs=("pair_id", "size"), aqua=("n_aqua_elig", "sum"), viirs=("n_viirs_elig", "sum")).to_string())
