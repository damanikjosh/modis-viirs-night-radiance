import pytest
import torch
from torch.utils.data import DataLoader

from nightrad.raw.synth import SyntheticTiles
from nightrad.raw.schema import split_batch


@pytest.fixture(scope="session")
def small_batch():
    b = next(iter(DataLoader(SyntheticTiles(8, 0, 48, 48), batch_size=8)))
    return split_batch(b)


