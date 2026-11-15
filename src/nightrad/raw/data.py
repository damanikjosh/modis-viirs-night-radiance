"""Loaders: packed .npy shards (memory-mapped, no per-epoch cache build), per-tile npz via the schema adapter, synthetic.

Pack once (`pack_dataset`) -> `<dir>/meta.json` + `<dir>/<key>.npy` per field, each [N, ...]; workers np.load(mmap_mode='r') lazily so
each worker only touches the pages of the tiles it reads (page cache shared across workers), no decompression, no per-epoch pass."""
from __future__ import annotations

import functools
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def pack_dataset(ds: Dataset, out: str, n: int | None = None):
    os.makedirs(out, exist_ok=True)
    n = n or len(ds)
    first = ds[0]
    mm = {k: np.lib.format.open_memmap(os.path.join(out, f"{k}.npy"), "w+", np.asarray(v).dtype, (n, *np.asarray(v).shape)) for k, v in first.items()}
    for i in range(n):
        t = ds[i]
        for k, v in t.items():
            mm[k][i] = v.numpy() if torch.is_tensor(v) else v
    for a in mm.values():
        a.flush()
    json.dump({"n": n, "keys": list(first)}, open(os.path.join(out, "meta.json"), "w"))


class PackedShards(Dataset):
    def __init__(self, path: str, idx: list[int] | None = None):
        self.path = path
        meta = json.load(open(os.path.join(path, "meta.json")))
        self.keys, self.idx = meta["keys"], (idx if idx is not None else list(range(meta["n"])))
        self._mm = None

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        if self._mm is None:  # opened lazily inside the worker process
            self._mm = {k: np.load(os.path.join(self.path, f"{k}.npy"), mmap_mode="r") for k in self.keys}
        j = self.idx[i]
        return {k: torch.from_numpy(np.array(a[j])) for k, a in self._mm.items()}


class NpzTiles(Dataset):
    """One .npz per tile (v3 layout), mapped through schema.adapt_v3_tile."""

    def __init__(self, files: list[str]):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        from nightrad.raw.schema import adapt_v3_tile
        with np.load(self.files[i], allow_pickle=False) as z:
            return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in adapt_v3_tile(z).items()}


def _winit(seed, wid):
    np.random.seed(seed + wid)


class FireBatchSampler(torch.utils.data.Sampler):
    """Batches that always hold fire tiles: `round(frac * batch_size)` items per batch are drawn from the tiles flagged `fire_flags`
    (the flag comes from the MODIS INPUT only, see modis_fire_flags), the rest from all tiles; both draws use the per-item `weights` (region-balanced when given), with replacement.
    One epoch = len(weights) // batch_size batches. Deterministic given `seed` and the epoch (reseed with set_epoch)."""

    def __init__(self, weights, fire_flags, batch_size: int, frac: float, seed: int = 0, n_items: int | None = None):
        self.w = torch.as_tensor(np.asarray(weights, dtype=np.float64)); self.fire = np.asarray(fire_flags, bool)
        self.bs, self.nf = batch_size, int(round(frac * batch_size)) if self.fire.any() else 0
        self.n_batches = (n_items or len(self.w)) // batch_size; self.seed, self.epoch = seed, 0
        self.wf = self.w * torch.as_tensor(self.fire.astype(np.float64))

    def set_epoch(self, e: int):
        self.epoch = e

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed * 1000003 + self.epoch)
        for _ in range(self.n_batches):
            a = torch.multinomial(self.wf, self.nf, replacement=True, generator=g) if self.nf else torch.zeros(0, dtype=torch.long)
            b = torch.multinomial(self.w, self.bs - self.nf, replacement=True, generator=g)
            yield torch.cat([a, b]).tolist()


def modis_fire_flags(ds, cfg_run, dev=None, bs: int = 32, level: float = 0.5, workers: int = 4) -> np.ndarray:
    """Per tile: does it hold >= 1 unsaturated, complete MODIS pixel whose own 4 um excess over its local median is >= `level` (ln units)? From the MODIS input only (never VIIRS, never a fire product)."""
    from nightrad.raw.core import Prep
    from nightrad.raw.schema import split_batch
    dev = dev or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = []
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers):
            inp, _ = split_batch(b, dev); P = Prep(inp, cfg_run)
            out.append(((P.ok_foot & (P.rho4["modis"] >= level) & (inp["modis_sat"] == 0)).flatten(1).any(1)).cpu().numpy())
    return np.concatenate(out)


def make_loader(ds, batch_size, workers=4, shuffle=False, seed=0, pin=True, drop_last=False, sample_weights=None, batch_sampler=None):
    g = torch.Generator().manual_seed(seed)
    if batch_sampler is not None:  # fire tiles in every batch (switch)
        return DataLoader(ds, batch_sampler=batch_sampler, num_workers=workers, pin_memory=pin and torch.cuda.is_available(),
                          persistent_workers=workers > 0, prefetch_factor=4 if workers > 0 else None, worker_init_fn=functools.partial(_winit, seed))
    if sample_weights is not None:  # region-balanced sampling (a switch): draw len(ds) items per epoch with these weights
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(torch.as_tensor(sample_weights), num_samples=len(ds), replacement=True, generator=g)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=workers, pin_memory=pin and torch.cuda.is_available(), drop_last=drop_last,
                          persistent_workers=workers > 0, prefetch_factor=4 if workers > 0 else None, worker_init_fn=functools.partial(_winit, seed))

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=pin and torch.cuda.is_available(),
                      generator=g, drop_last=drop_last, persistent_workers=workers > 0, prefetch_factor=4 if workers > 0 else None,
                      worker_init_fn=functools.partial(_winit, seed))


class CudaPrefetcher:
    """Wraps a DataLoader: copies the NEXT batch to the GPU on a side stream while the current step runs (pinned memory, non_blocking)."""

    def __init__(self, loader, device):
        self.loader, self.device = loader, device

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        stream = torch.cuda.Stream()
        it = iter(self.loader)

        def load():
            try:
                b = next(it)
            except StopIteration:
                return None
            with torch.cuda.stream(stream):
                return {k: v.to(self.device, non_blocking=True) for k, v in b.items()}

        nxt = load()
        while nxt is not None:
            torch.cuda.current_stream().wait_stream(stream)
            cur = nxt
            for v in cur.values():
                v.record_stream(torch.cuda.current_stream())
            nxt = load()
            yield cur



class CombinedShards(Dataset):
    """Several packed night sets as one training set (old v3 night + night-fire tiles). Only the keys present in ALL sets are returned (so eval_* extras that exist in one pack do not break collation);
    every item also carries `eval_source` (index of the set, int32) and `eval_region` (int32; sets that have no region labels use -1 - 1000*source... i.e. one pseudo-region per set)."""

    def __init__(self, paths: list[str]):
        self.sets = [PackedShards(p) for p in paths]
        common = set(self.sets[0].keys)
        for d in self.sets[1:]:
            common &= set(d.keys)
        self.keys = [k for k in self.sets[0].keys if k in common and k not in ("eval_region", "eval_episode", "eval_far_km")]
        self.index = [(si, i) for si, d in enumerate(self.sets) for i in range(len(d))]
        self._regions = None

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        si, j = self.index[i]
        it = self.sets[si][j]
        return {k: it[k] for k in self.keys}

    def region_ids(self) -> np.ndarray:
        """Per-item region id for balanced sampling: the packed eval_region where it exists, else one pseudo-region per set (-1000 - source)."""
        if self._regions is None:
            out = []
            for si, d in enumerate(self.sets):
                f = os.path.join(d.path, "eval_region.npy")
                out.append(np.load(f).reshape(len(d), -1)[:, 0].astype(np.int64) if os.path.exists(f) else np.full(len(d), -1000 - si, np.int64))
            self._regions = np.concatenate(out)
        return self._regions


def region_balanced_weights(region_ids: np.ndarray, cap_share: float = 0.2) -> np.ndarray:
    """Per-item sampling weights so every region contributes equally per epoch, with no region above `cap_share` of the draws (cap applied after equalising when there are > 1/cap regions)."""
    ids, inv, cnt = np.unique(region_ids, return_inverse=True, return_counts=True)
    w_region = np.full(len(ids), 1.0 / len(ids))
    return (w_region[inv] / cnt[inv]).astype(np.float64)
