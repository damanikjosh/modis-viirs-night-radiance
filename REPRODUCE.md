# Reproducing the paper and the layer

Hardware: everything marked CPU runs on a laptop. Full model training (one run: 46 epochs of 647 steps) needs one GPU (about 2.7 h on an RTX 4090 per the paper; the whole study used about 60 GPU-hours) and the packed training data (about 86 GB of NASA granules for the 1,017 pairs, then 32,914 tiles).
Status words: **re-run** = executed from this repository and compared with the stored numbers; **runs** = the command was executed (smoke scale or one case) but not at paper scale; **not re-run** = documented, not executed here.
`$D` = `$NR_DATA_ROOT`. Detailed commands: `docs/paper.md`, `docs/model.md`, `docs/data.md`, `docs/layer.md`.

## A. Paper results

| Paper result | Command | Inputs | Expected output / tolerance | Runtime / hardware | Status |
|---|---|---|---|---|---|
| Stored results are the original files | `python scripts/paper/verify_manifest.py` | `results/` (121 files) | all sha256 match `results/MANIFEST.json` | seconds, CPU | re-run |
| All number macros (text, Tab. 1, ablations, closed test), 18 tables | `python scripts/paper/gen_numbers.py --out out/paper` | `results/` | 1,204 of 1,204 macros and 18 of 18 tables identical to the generated files used by the paper (`tests/paper/expected`) | seconds, CPU | re-run (59 AP/FSS/skill macros come from `metrics2_summary.json`: its hash is checked, values are not re-derived; 54 macros are recorded values with their source) |
| Figures made from stored results | `for f in scripts/paper/figures/make_*.py; do python $f --out out/paper/figures; done` | `results/` | 15 of 15 identical to the committed PDFs (rasterised, 0 differing pixels); two earlier-version figures differ, see `docs/paper.md` | seconds, CPU | re-run |
| Experiment A (seed separation, A1-A4) | `python scripts/exp/a_stored.py results out/exp_a` | `results/` | counts 8/4/2/0/2 of 10 | seconds, CPU | re-run |
| Pairing, tiling, funnel (Sec. 4, Tab. counts) | `docs/data.md` pipeline: `find_pairs`, `select_pairs`, `download`, `build_tiles`, `finalize`, `make_nf2_split`, `pack` | NASA granules, FIRMS | one pair (18 Oct 2019, Java) rebuilt from public NASA files: 5 of 5 stored tiles, 360 arrays, integer arrays exact, floats within 1e-4. Frozen list of the 1,017 pairs is shipped (`src/nightrad/data/resources/pairs_paper.csv`) | one pair: minutes, CPU; all pairs: about 86 GB download, GPU optional | runs (one pair); full 1,017-pair build **not re-run** |
| Held-out episode data (Kalimantan 2015) | `episode_index.py`, `build_episode.py`, `episode_finalize.py`, `episode_pack.py` | NASA granules | 697 night tiles, radiance arrays identical to the stored pack | hours, CPU | re-run (data build only) |
| Model, losses, sensor views, gates (Sec. 5-6) | `uv run pytest tests/raw` ; `scripts/diag_gates.py` | synthetic tiles / packs | unit tests; gates on real tiles need the packs | tests: minutes CPU | tests re-run; gates on real tiles **not re-run** |
| Training of the reported runs (Tab. 1, runs list) | `scripts/train_raw.py --preset gammabw100-nf2 --seed 0 ...` (presets in `configs/paper_runs.json`) | packed night training data | about 29,762 steps per run; results differ at the level of the seed spread (not bit-exact across hardware) | 2.7 h, 1 GPU per run | **not re-run**; synthetic smoke training runs on CPU (`docs/model.md`) |
| SELECT scoring, pick rule (Tab. 1, Sec. 7) | `scripts/score_run.py`, `scripts/apply_pick_rule.py` | runs, SELECT pack | the stored score files in `results/raw/select` are what the paper reports | minutes per run, GPU | scorer code runs on synthetic data; real-data scoring **not re-run** |
| Post-hoc measures (AP, FSS, CSI, skill) | `scripts/descriptive_score.py`, `descriptive_measures.py` | runs, SELECT pack | descriptive CSV in `results/raw/_descriptive` | minutes | measures from stored files re-run; re-scoring **not re-run** |
| Held-out episode (Tab. closed) | `scripts/closed_test.py` (dry run by default; `--execute` only with the frozen candidate) | held-out pack (not included) | scored **once**: 1 of 4 pre-stated lines passed, verdict fail. **Not a command to repeat and tune on.** The stored verdicts regenerate the table | - | verdict regeneration re-run; the scoring itself **not re-run** |
| Ablations: loss family, brightness weight, bound, best vs final, width, window, scan angle, 4 µm-only (Tab. abl) | `docs/paper.md` table (`scripts/exp`, `scripts/design`, `scripts/sweep1_driver.py`) | packs | rows computed from stored scores are re-run (see above); experiments C, D, B5-B8 and the 4 µm-only row need packs and GPU | hours-days, GPU | stored numbers regenerated; experiments **not re-run** |
| Pairing and architecture figures | none | orbit data, drawings | - | - | **not included** |

## B. Harmonized VIIRS layer (all CPU)
Steps and options: `docs/layer.md`. Checkpoint: `weights/best_val.pt` (sha256 `87251b2bb8f151d3e941922862f5f146d7b6e0dff02f519d337ab6e8e523dc26`, enforced).

| Step | Command | Expected | Status |
|---|---|---|---|
| 1 granule list | `scripts/layer/catalogue.py --bbox 94.5,-11.5,141.5,6.5 --start 2006-08-05 --end 2006-08-06 --ids A2006217.1835,... --out $D/cat.jsonl` | one JSON line per granule (NASA CMR) | re-run |
| 2 granule -> cells | `scripts/layer/worker.py --catalogue $D/cat.jsonl --bbox ... --region indonesia` | per granule: counts and input sha256 as in the product record | re-run |
| 3 monthly cells | `scripts/layer/aggregate.py --catalogue $D/cat.jsonl` | de-duplicated cells, coverage | re-run |
| 4 display points | `scripts/layer/points.py --only-complete`, `check_points.py` | one point per MODIS pixel and night | re-run |

Comparison with the stored product rows, 6 Aqua night granules (August 2006, Sumatra/Kalimantan; about 600 MB, 92 s worker time on a laptop CPU): 2,509 flagged cells = 2,509 stored, identical cell sets; predicted radiance differs by at most 9e-4 (0.026 K); 1,380 display points = 1,380 stored, positions within 0.29 m; independent re-count 0 failures. Only 2006-08 was compared; seam duplicates (not present in the sample) are unit-tested only; no GPU run (CPU and GPU agree on cell sets in tests, not bit-wise).

## C. Tests
`uv sync && uv run pytest` on a fresh clone.
