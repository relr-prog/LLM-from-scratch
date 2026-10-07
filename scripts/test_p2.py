import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
from llm.config import ModelConfig, TrainConfig
from llm.mmap_dataset import MMapTokenDataset
from llm.model import GPT

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "tokens.bin"
    np.arange(4096, dtype=np.int32).tofile(path)
    ds = MMapTokenDataset(path, block_size=32, start=0, end=4096)
    x, y = ds[10]
    assert x.shape == (32,) and y.shape == (32,)
    assert torch.equal(y[:-1], x[1:])

cfg = ModelConfig(
    vocab_size=512,
    block_size=32,
    n_layer=2,
    n_head=4,
    n_embd=128,
    gradient_checkpointing=True,
)
model = GPT(cfg)
idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
_, loss = model(idx, idx)
loss.backward()
assert torch.isfinite(loss)
assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())

train_cfg = TrainConfig(distributed="auto")
assert train_cfg.keep_checkpoints == 3
print("p2 smoke: ok")
