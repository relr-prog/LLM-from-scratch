from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class MMapTokenDataset(Dataset):
    """Random-access next-token dataset backed by an int32 mmap file."""
    def __init__(self, path, block_size, start=0, end=None):
        self.path = str(path)
        self.block_size = int(block_size)
        size = Path(path).stat().st_size // 4
        self.start = int(start)
        self.end = size if end is None else min(int(end), size)
        if self.start < 0 or self.end > size or self.start >= self.end:
            raise ValueError("invalid dataset range")
        if self.end - self.start <= self.block_size:
            raise ValueError("dataset range is too small for block_size")

        self._data = None

    def _open(self):
        if self._data is None:
            self._data = np.memmap(self.path, mode="r", dtype=np.int32)

    def __len__(self):
        return self.end - self.start - self.block_size

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        self._open()
        i = self.start + index
        x = torch.from_numpy(self._data[i:i + self.block_size].copy()).long()
        y = torch.from_numpy(self._data[i + 1:i + self.block_size + 1].copy()).long()
        return x, y
