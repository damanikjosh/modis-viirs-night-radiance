# Torch tile builder (`build_tiles.py --device cuda | torch-cpu`)

Same tile schema (v5) and same selection logic as the CPU reference; the numeric work runs in torch on the device.
`--device cpu` (default) stays the reference (numpy / scipy, one process per pair).

    uv run python scripts/build_tiles.py --pairs paper --device cuda --workers 6 --static STATIC.csv --exclusions paper --download

`--workers` = reader processes (files -> small numpy arrays; at most workers + 1 pairs are read ahead, which bounds RAM); two writer threads validate and
`savez_compressed`. `--device torch-cpu` runs the same torch code on the CPU (used by the tests and the smoke run in `docs/data.md`).

## Where the work goes
| stage | where | notes |
|---|---|---|
| HDF4/NetCDF reads, MODIS L1B BT, fire-mask decode | CPU reader processes | VIIRS: raw 750 m geolocation, raw uint16 I4/I5, LUTs only (`readers_gpu.read_viirs_raw`) |
| VIIRS 2x per-scan upsampling, LUT -> BT, class map | device | `readers_gpu.viirs_gswath`; bit-identical to `readers.read_viirs` |
| coverage test (coarse lattice), window search, LAEA projection, pixel size, nearest pixel (K=3 + nadir pick), cloud-adjacency, static, 21x21 background z-scores, local-median fill | device, batched over 8 tiles | `tiles_gpu.build_tiles_gpu` |
| candidate selection (fire centres, 24 km thinning, random tiles, overlap test, sample weights) | CPU, replays the CPU loop exactly; tile builds are speculative batches | `tiles_gpu.build_pair_tiles_gpu` |
| validate + npz | writer threads | |

Equivalence (`tests/test_tiles_gpu.py`, `tests/test_v2_tiles.py`; the smoke run of `docs/data.md` rebuilds one pair on both paths): all integer/flag layers
(cls, pid, flags, overlap, ...) identical; float layers within 2e-5 (scan angle, FRP: float32 libm differences).
Documented limits: nearest-pixel ties closer than ~1e-9 km could resolve differently (shortlist is re-ranked in fp64);
VIIRS window crop radius is 40 km (CPU: 80 km), identical in the tile area (the 21x21 background needs 10 pixels = <=8 km).
GPU memory: ~3 GiB peak per pair (measured), capped by `--gpu-mem-gb`.
