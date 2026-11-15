# Model, losses, training and scoring

Code: `src/nightrad/raw/` (package `nightrad.raw`), commands: `scripts/` (listed with their paper results in `docs/entrypoints/model.txt`).
The model reads the radiance image of ONE sensor per forward pass (MODIS or VIIRS) and returns a scene on the 375 m grid; the other sensor enters only through a loss, in training.

What the model takes from the fire products: no fire detections (class, power, confidence) are an input or a target. The products give a cloud / water / glint / bow-tie
class and the product-declared saturated VIIRS cells, which form the validity mask (`raw/mask.py`, `raw/schema.py`: the mask gates the input and decides which cells every
loss term scores), and they choose scenes, tile centres and sampling weights when the data are built. Fire level in the loss weights, the fire-batch sampler and the scores is
defined from measured radiance (MODIS input only).

## Architecture (`raw/net.py`, `raw/layers.py`)
Input: 12 channels on the 96 x 96 grid of 375 m cells (`net.build_features`: scaled log radiance of both bands against a 300 K Planck reference, log excess over the local
masked median, mask, saturation, MODIS 21 band, view angle, pixel size, solar zenith). Sensor-specific Embedding (two masked 3x3 convolutions, 32 channels, FiLM) ->
shared three-level masked U-Net Encoder (32/64/96 channels, two skips, 1x1 to a 16-channel latent) -> shared Decoder (two blocks, one dilation 2, zero-initialised 1x1 head,
4 outputs per cell). All convolutions are partial (mask-normalised), so a value outside the mask cannot influence an output inside it (tested). 438,678 parameters: 2 gains,
2 x 13,248 Embedding, 379,536 Encoder, 32,644 Decoder.

Scene: `ln B_b = ln A_b + delta_max tanh d_b`, `s_b = min(softplus(o_b) + g_b rho_b^+, s_max)`, `E_b = B_b (exp(s_b) - 1)`; `A_b` is the local masked median anchor
(7x7 taps, spacing 2). Fixed views (`raw/views.py`): VIIRS view `B + E`; MODIS view = footprint-weighted background converted by Planck plus `f_b` times the footprint-weighted
excess (`t_fire` 800 K).

Options kept because they are used by reported runs: `arch conv3` (three-convolution network of the loss-family sweep), `width`/`latent` (width gate), `delta_max` 0.25/0.5,
`out4` (see below), the allocation head (`alloc`, used only by the placement probe). The block, allocation and hurdle terms of the full objective exist in
`raw/core.py` and are off in every reported run, as the paper states.

## Losses (`raw/losses.py`, `raw/core.py`)
Four terms with fixed weights (cross-sensor M->V 1.0, V->M 0.5, same-sensor hidden M->M 0.1 and V->V 0.1), evaluated inside the mask only; hidden cells weigh 1.0, visible 0.25 (`raw/hide.py`).
`--loss legacy` is the Group-share loss (Huber 0.1 on ln radiance, shares 0.4/0.2/0.4 for background / near-fire / fire-level cells, groups from the target radiance).
`--loss gamma` is the Gamma deviance on linear radiance at three scales (cell, MODIS pixel, 3x3 block), `--loss-family poisson|nmse` the other deviances of the sweep,
`--w-energy 0.1` the energy term, `--bright-weight G` the Gamma-BW weight `1 + G clamp(rho4_modis, 0, 3)` (MODIS input only). Saturated MODIS pixels are one-sided.

## Training (`scripts/train_raw.py`, `raw/train.py`)
AdamW, peak rate 2e-3 (constant after 100 warm-up steps with `--lr-const`), weight decay 1e-4, batch 32 (a quarter drawn from tiles with a fire-level MODIS pixel,
`--fire-batch-frac 0.25`), 46 epochs, bf16 trunk with fp32 radiometry, clip 1.0, hiding windows 9 x 9 (probability 0.5, 3 on the brightest fire-level cells, 1 random, jitter 3).
Per epoch the full-validation loss on the SELECT tiles of `--val-list configs/nf2_split.json` is written to `diag.jsonl`; `best_val.pt` is the lowest, `ep<N>.pt` every epoch.
Data are packed shard directories (`nightrad.raw.data.pack_dataset`); give them as arguments (shell convenience: `$NR_DATA_ROOT`). `--val-select <split JSON>` is the older way to
choose the validation tiles by episode list (`configs/val2_split.json`, `configs/nf2_split_episodes.json`); `--val-list` chooses by tile id.

Presets (`configs/paper_runs.json`, flags read from the stored `run_meta.json` of the runs): `gammabw100-nf2`, `gammabw1000-nf2` (seeds 0-3), `gamma-nf2`, `gammabw100-d05-nf2`,
`gammabw1000-d05-nf2` (bound 0.5), `oldloss-nf2` (Group-share, no energy term). `train_raw.py --preset NAME --seed S --data ... --val-data ... --sample-weights ... --out runs/NAME-sS`.

| setting | value |
|---|---|
| optimiser / peak rate / weight decay | AdamW / 2e-3 / 1e-4 |
| warm-up / schedule | 100 steps / constant |
| batch / epochs | 32 / 46 (Gamma and Group-share runs stopped by hand at 20 / 32) |
| width / latent | 32 / 16 |
| delta_max / s_max / s bias | 0.25 (0.5 for -d05) / 8 / -4 |
| loss weights mv / vm / mm / vv / energy | 1 / 0.5 / 0.1 / 0.1 / 0.1 |
| hide window / p / top / jitter | 9 / 0.5 / 3 / 3 |
| min valid fraction / widen | 0.10 / 0 |
| T_fire / kernel | 800 K / tri (w_scan 0.001) |

## Stage-1 gates (`scripts/diag_gates.py`)
`--gate overfit` (memorise 8 tiles, hiding off; loss fall > 10x, error at VIIRS fire cells < 0.1, energy at MODIS-bright cells 0.9-1.1), `path` (training and evaluation path give the same loss),
`seed` (two identical short runs), `share` (loss share of bright cells), `bf16`. `--delta-max 0` is the no-background-correction row; `--width W --latent L` the width rows.

## Scoring (frozen protocol in `configs/protocol_eval_v2*.json`)
`train_const.py` computes the input-blind constant once; `score_run.py --set select` writes `score_select.json` (block AUC, energy, hottest-cell share, flags, amount measure, with and
without blocks next to masked holes) and `--baselines-from` the non-learning references; `--dt-range LO HI` scores only tiles inside a pairing-window bin. The scorer refuses
tiles of the REPORT list and of `configs/report_retired.json` (guard on tile ids, `raw/nf2_split.py`). `apply_pick_rule.py` applies `configs/model_pick_rule.json`.
`closed_test.py` scores ONE frozen candidate (`configs/closed_test_candidates.json`) once on the held-out episode under the approved amendment
(`configs/closed_test_amendment.json`), then `closed_test_reader.py` turns the scores into the four lines (paper: 1 of 4 passed). Post-hoc measures (marked in the paper):
`descriptive_score.py`, `descriptive_label.py`, `descriptive_measures.py`, checked by `descriptive_crosscheck.py`.

Pinned files: the sha256 of `configs/nf2_split.json` and `configs/report_retired.json` are pinned in `raw/nf2_split.py`. The paper's files (recorded in its score files, 6851bb72... and
e41d5155...) differ from the committed ones only by free-text strings; tile lists are identical.

## 4 um-only output (paper ablation table)
`train_raw.py --out-bands 4` (`NetConfig.out4`): decoder, scene, views and every term cover only the 4 um band; inputs stay both bands. At scoring the rule needs an 11 um radiance,
which `raw/c11.py` takes from the MODIS 11 um measurement by Planck conversion (no VIIRS value). The change was developed on a separate branch of the original development repository, based on an earlier code state; it was applied to
this code with its tests (`tests/raw/test_out4.py`, golden values for the default path). The ablation runs themselves were trained on that branch; this repository reproduces the option, not those exact runs.

## Loss-family sweep and placement probe
`configs/lossfamily_sweep.json` (24 runs: Group-share default, Gamma / Poisson / squared-error deviance; U-Net and three-convolution network; seeds 0-2), driven by
`sweep1_driver.py`, halves A/B in `configs/lossfamily_halves.json`, re-scored by `protocol_eval.py`. The per-run rows table of the paper was assembled outside this repository.
`placement_probe.py` trains the placement probe.

## Open points
Options not used by any reported run remain in `train_raw.py` / `raw/` (collapse stop, region balance, sun term, hurdle/measured modes, tie11, partial hiding, input weight, false share),
because the paper states that the block, allocation and hurdle terms exist and are off.

## Pre-stated protocol files (paper appendix: protocol order)
The scorer reads the frozen protocol `configs/protocol_eval_v2.json` and its dated amendments, kept unchanged as the record of what was fixed before which scores:
`configs/protocol_eval_v2_amendment1.json`, `configs/protocol_eval_v2_amendment2.json`, `configs/protocol_eval_v2_amendment4.json`, `configs/protocol_eval_v2_amendment5.json` and `configs/protocol_eval_v2_amendment5_memorise_gate.json` (the Stage-1 memorise gate).
