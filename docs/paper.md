# Regenerating the paper's numbers, tables and figures

Paper: *Learning VIIRS Night Radiance from MODIS without Fire Products*. Everything that can be recomputed from stored score files is
recomputed from `results/` (18 MB, sha256-checked) on a laptop CPU in seconds; nothing here needs a GPU, satellite data or network access.

```
python scripts/paper/verify_manifest.py                 # 121 stored files match results/MANIFEST.json
python scripts/paper/gen_numbers.py --out out/paper     # numbers.tex (1,204 macros), 18 tab_*.tex, NUMBERS.md (number map)
for f in scripts/paper/figures/make_*.py; do python $f --out out/paper/figures; done
python scripts/exp/a_stored.py results out/exp_a        # experiment A1-A4 from stored scores
pytest tests/paper tests/design                         # all of the above, compared with the paper's own generated files (tests/paper/expected)
```

## What is stored (`results/`)

| path | content | provenance |
|---|---|---|
| `raw/select/<run>/score_select.json` (29) | scored SELECT result of every run and of the simple methods (`baselines`) | copied unchanged |
| `raw/old/<run>/score_oldval.json` (25) | same on the old 9-pair night validation | copied unchanged |
| `raw/<run>/run_meta.json`, `log.jsonl` (16 runs) | run record (arguments, config, pack report) and per-epoch training log | copied unchanged |
| `raw/_descriptive/*.csv` | post hoc descriptive measures, calibration bins | copied unchanged |
| `raw/placement/placement.json`, `raw/sweep1/rows.jsonl`, `raw/wide/*_3k.json` | hottest-cell placement probe, loss-family sweep rows, width test | copied unchanged |
| `raw/_closed_test/{score_night,verdict_night,verdict_day}.json` | held-out episode: line verdicts and score aggregates (see below) | copied unchanged |
| `raw/_closed_test/closed_test_record.json`, `raw/_closed_test_prep/pick.json` | the two verdicts / the pick-rule outcome | **derived** (fields that are not needed for any number are dropped; manifest says `derived`, `source_sha256` is the hash of the original) |
| `metrics2_summary.json` | post hoc block AP, FSS, hottest-cell exact rate, amount skill per run | compacted by `scripts/paper/extract_metrics2.py` from per-run files that are not stored here |
| `static_numbers.tsv`, `static_tables/*.tsv` | recorded values that no stored score file holds (data counts, experiment results, compute figures), each with its source | typed from the project records (source column reworded to name the record or experiment, no internal paths); **not recomputed** |
| `design/*` | recorded outputs of the design tests (`scripts/design/`) on TRAIN and SELECT tiles | copied; the literature column of `front_spread.csv` reworded |

Held-out episode: `_closed_test/` holds only aggregates (line verdicts, per-scene-pair AUC rows of the four night pairs, calibration bins, counts). No per-tile or
per-pixel prediction and none of the held-out tile files are included; `tests/paper` asserts that no list in these files is longer than 22 items. Nothing from the retired set is stored.

`scripts/paper/make_manifest.py` rewrites the manifest after a deliberate change; `verify_manifest.py` (also run inside `gen_numbers.py`) fails on any changed, missing or unlisted file.

## Paper item -> script -> stored input -> regenerated

Legend: **yes/exact** = regenerated here and identical to the paper's generated file; **recorded** = a typed value with its source, not recomputed; **not included** = needs data or tooling that this repository does not hold.

| paper item | script | stored input | regenerated? |
|---|---|---|---|
| Abstract, Sec. 1-3, 5-6, 10-12 numbers (about 60 macros) | `gen_numbers.py` | `raw/select`, `raw/wide`, `raw/_closed_test`, `metrics2_summary.json`, `static_numbers.tsv` | yes/exact (computed ones); recorded (`static_numbers.tsv`) |
| Tab. 1 (`tab:main`, 99 macros: block AUC on/off, AP, energy, flags, error, amount skill, hottest share, exact rate; baselines; chance) | `gen_numbers.py` (macros), `paper/sections/tab_main.tex` is the layout | `raw/select/*/score_select.json` (62 macros), `metrics2_summary.json` (37) | yes/exact (macro values); AP/FSS/skill rest on `metrics2_summary.json`, whose per-run inputs are not stored (checked by sha256 only) |
| Tab. `tab:lines` (pre-stated lines) | `gen_numbers.py` | constants `E_LO ... L1_SHARE` copied from `configs/model_pick_rule.json`, `configs/protocol_eval_v2_amendment4.json` | yes/exact |
| Pick-rule outcome (Sec. 7) | `gen_numbers.py` (`Pk*`, `Pick*`) | `raw/_closed_test_prep/pick.json`, `raw/select` | yes/exact |
| Tab. `tab:closed` (held-out episode, 31 macros) and the abstract/intro/discussion statements about it | `gen_numbers.py` (`Ct*`) | `raw/_closed_test/*` | yes/exact (reads the stored verdicts; the held-out scoring itself is not re-run) |
| Tab. `tab:counts` (data funnel) | `gen_numbers.py` + `static_tables/counts.tsv` | `raw/gammabw1000-nf2-s0/run_meta.json` (pack report), `static_numbers.tsv` | partly: pack counts yes/exact; the other counts recorded |
| Tab. `tab:abl` rows loss family | `gen_numbers.py` | `raw/sweep1/rows.jsonl` | yes/exact |
| `tab:abl` rows brightness weight, bound, best/final, small list vs large data | `gen_numbers.py`; `scripts/exp/a_stored.py` (A1-A4) | `raw/select`, `raw/old`, `raw/*/log.jsonl`, `raw/_descriptive` | yes/exact; the six A1 seed-separation counts (`ExpSep*`) are reproduced by `a_stored.py` (8/4/2/0/2 of 10 pairs) |
| `tab:abl` row network width | `gen_numbers.py`, `figures/make_width.py` | `raw/wide/*_3k.json` | yes/exact |
| `tab:abl` rows no background correction; energy and hidden-region terms (experiment C) | `scripts/exp/c_run.sh`, `c_gates.sh`, `c_score.sh`, `c_queue.sh`, `c_gate_eval.py`, `c_analysis.py` | recorded in `static_numbers.tsv` (`NoBgFall`, `ShortSteps`, ...) and `static_tables/shortbudget.tsv` | recorded; experiment needs packs and a GPU, not re-run |
| `tab:abl` row 4 um-only output | none in this repository | `static_tables/fourmicron.tsv`, `static_numbers.tsv` | recorded; the 4 um-only variant lives on a code branch that is not part of this repository |
| `tab:abl` row pairing window (experiment D) | `scripts/exp/d_dist.py`, `d_subsets.py`, `d_run.sh`, `d_score.sh`, `d_analysis.py`, `d_dist_fig.py` | recorded in `static_numbers.tsv` (`Win*`) | recorded; not re-run |
| `tab:abl` row scan angle (experiments B5-B7), inference cost (B8) | `scripts/exp/extract_select.py`, `run_collect.py`, `common.py`, `b_analysis.py`, `inference_cost.py` | recorded in `static_numbers.tsv` (`Exp*`, `ModelMB`, `Inf*`) | recorded; not re-run |
| Design tests (time-window size, agreement vs gap, scan-angle distribution, model error vs gap) | `scripts/design/{extract_train,collect,analyze_dt,scan_dist,size_vs_threshold}.py`, `run_design.sh` | recorded outputs in `design/`; summary numbers in `static_numbers.tsv` | figures from the recorded outputs: yes/exact (5 of 5); the collection step needs tiles and is not re-run (`analyze_dt.py` is tested on a synthetic planted effect in `tests/design`) |
| Tab. `tab:runsall` (appendix, per-run values) | `gen_numbers.py` -> `tab_runs_all.tex` | `raw/select/*/score_select.json` | yes/exact |
| Tab. `tab:hyper` | `gen_numbers.py` -> `tab_hyper.tex` | `raw/gammabw1000-nf2-s0/run_meta.json` | yes/exact |
| Tab. `tab:variants` | `gen_numbers.py` -> `tab_variants.tex` | `raw/select` | yes/exact |
| Tab. `tab:priorwork` | none (text) | cited papers | not a result |
| Fig. `fig:pair-hist`, `fig:pair-date`, `fig:trajectory` (pairing) | pairing figure scripts of the paper repository | need granule overlaps and orbit (TLE) data | **not included** |
| Fig. `fig:arch`, `fig:net` (architecture) | TikZ sources | none | **not included** (drawings, no data) |

Tables generated but not used in the paper text (kept because the generator writes them from the same inputs): `tab_decision`, `tab_groups`, `tab_baselines`, `tab_pick`, `tab_desc`, `tab_runlist`, `tab_lossfamily`, `tab_width`, `tab_metrics2`, `tab_metrics2_off`, and the static `tab_counts`, `tab_gates`, `tab_regions`, `tab_fourmicron`, `tab_shortbudget`.
Figures from stored results that the current paper PDF does not include (earlier versions did): `scripts/paper/figures/make_{auc,amount,lines,desc,calib,fireshare,curves,lossfamily,placement,width}.py` -> `fig_*.pdf`, and `make_design_figs.py` -> `fig_design_*.pdf/png`.
Left out on purpose: the LaTeX build, paper text, landing-page generator, result fetching, the consistency checker of the paper repository, and the project timeline figure.

## Re-run on the Mac (CPU, `nice -n 15`, one process)

* Numbers: 1,204 of 1,204 macros recomputed, identical values: identical values (of them 296 are used by the paper text: 241 computed from stored files or the pre-stated constants, 1 derived from two recorded values, 54 recorded). The only textual difference in `numbers.tex` is the generator comment on line 1.
* Tables: 18 of 18 generated `tab_*.tex` byte-identical to the paper's committed files (13 computed, 5 from recorded values).
* Number map: 1,204 of 1,204 values equal to the paper's `NUMBERS.md`; the source column differs by design (stored paths instead of URIs, neutral wording for recorded sources).
* Figures: 10 of 10 `fig_*.pdf` and 5 of 5 design figures (PDF and PNG) identical to the paper's committed figures after rendering (same text, 0 differing pixels).
* Experiment A: A1 seed-separation counts match the recorded ones (6 of 6). `a3` and `a4` figures: 0 differing pixels. `a1`: 603 pixels differ in one row of markers (the committed figure predates seeds 2 and 3 of the brightness-weighted settings, which the stored scores now contain). `a2`: the committed figure has a point at epoch 0 (reading before training) that the stored training log does not contain; all other points coincide (epochs are shifted by one because the log counts from 0).

## Known limits

* The `Pk*` and `Ct*` numbers rest on the stored aggregates; the held-out scoring is not re-run here by design.
* 59 of the 296 used macros come from `metrics2_summary.json`, a stored compaction whose per-run inputs are not stored.
* `scripts/paper/figures/common.py` keeps its own copy of the line logic (`generator` cannot be imported because it runs on import); `tests/paper` checks that both agree.
* Experiment runners `c_run.sh` and `d_run.sh` call `scripts/train_raw.py --val-select ...`, an option that the current `scripts/train_raw.py` does not have (it has `--val-list`); they document the procedure that produced the recorded numbers and need that option before they can be re-run.
