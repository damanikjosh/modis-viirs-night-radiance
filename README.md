# modis-viirs-night-radiance

Code for **"Learning VIIRS Night Radiance from MODIS without Fire Products: Near-Simultaneous Pairs, a Shared Latent Scene, and a Pre-Stated Evaluation"** (Joshua Julian Damanik, technical report) and for the **Harmonized VIIRS** fire layer built with it.

A small network reads the 4 and 11 µm radiance of one Aqua-MODIS night scene and predicts what S-NPP-VIIRS (375 m) would have measured over the same ground. It is trained on near-simultaneous Aqua x S-NPP scene pairs (30 minutes or less). The Harmonized VIIRS layer applies the trained model to every Aqua night granule of 2002-2011 and keeps the cells where the predicted VIIRS-view radiance passes a documented flag rule.

## What the model takes from NASA fire products (exact)
- **No fire detection is an input or a target**: no fire class, no fire radiative power, no detection confidence reaches the network or the loss.
- The fire-product files (MYD14, VNP14IMG, VJ114IMG) are read for their **cloud, water, glint, bow-tie and quality classes**, which form the validity mask that gates the input and the loss, and VIIRS cells at the product's 367 K saturation fill value are removed.
- Fire detections are used to **choose** scenes, nights and tile centres of the training data and for sampling weights and validation-set balance, and as report-only arrays in evaluation.
- The input of the model is MODIS (MYD021KM) radiance and geometry (MYD03). At inference no VIIRS data is needed, but a clear-land mask from the fire product's classes is.

## Status of the results (read this first)
- The model is **experimental**. On the selection set it meets three of the pre-stated lines for one setting; it is optimistic by construction.
- The **held-out episode (Kalimantan, September 2015) was scored once**, on one frozen candidate (Gamma-BW100, seed 0, best-validation checkpoint): **1 of 4 pre-stated lines passed**, the verdict is **fail** (see the paper, Table "closed"). The held-out scorer is included as code for transparency. It is not a thing to re-run and tune on; this repository does not contain the held-out tiles or predictions.
- The Harmonized VIIRS layer is a model output, not a NASA product; night and Aqua only.

## Install and test
```
uv sync
uv run pytest
```
Python 3.12, CPU is enough for the tests, the paper regeneration, the smoke runs and the layer on a few granules. Training the reported models needs a GPU (about 2.7 h per run on one RTX 4090, per the paper).

## Use
| I want to | read |
|---|---|
| regenerate every number, table and figure of the paper from stored score files | `docs/paper.md`, `REPRODUCE.md` |
| run the layer (granule list -> model -> monthly cells -> display points) | `docs/layer.md` |
| build the training data (pairs, download, tiles, splits, packs) | `docs/data.md` |
| train, gate, score and pick models, descriptive measures | `docs/model.md` |
| see which command produces which paper result, with its re-run status | `REPRODUCE.md` |

NASA input data are not redistributed. Downloads use a NASA Earthdata account: set `NR_EARTHDATA_TOKEN` (bearer token) or put the login in `~/.netrc`; set `NR_DATA_ROOT` for the data directory. Nothing secret is stored in the repository.

## Model weights
`weights/best_val.pt` (1.7 MB): Gamma-BW100, seed 0, best-validation checkpoint, sha256 `87251b2bb8f151d3e941922862f5f146d7b6e0dff02f519d337ab6e8e523dc26`. The layer worker refuses any other file.

## Data sources
Aqua MODIS MYD021KM, MYD03, MYD14 (collection 6.1); S-NPP VIIRS VNP02IMG, VNP03MOD, VNP14IMG (version 2 / 002); NOAA-20 VJ114IMG; all from NASA LAADS DAAC / LP DAAC; granule lists from NASA CMR; fire detections for scene selection from the NASA FIRMS archive. See `docs/data.md` for the exact versions.

## Licence
Code and the included model weights: Apache-2.0 (`LICENSE`, `NOTICE`). NASA input data are public-domain open data under NASA's data policy and are not redistributed here. The paper text is not part of this repository.

## Citation
See `CITATION.cff`.
