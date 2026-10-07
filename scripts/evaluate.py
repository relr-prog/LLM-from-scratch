import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "src")
from llm.config import ModelConfig
from llm.mmap_dataset import MMapTokenDataset
from llm.model import GPT

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default="checkpoints/latest.pt")
ap.add_argument("--data", default="data/train.bin")
ap.add_argument("--split", type=float, default=0.9)
ap.add_argument("--steps", type=int, default=100)
ap.add_argument("--batch-size", type=int, default=2)
args = ap.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
cfg = ModelConfig(**ckpt["model_config"])
model = GPT(cfg).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

size = Path(args.data).stat().st_size // 4
split = int(size * args.split)
dataset = MMapTokenDataset(args.data, cfg.block_size, split, size)
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

total = 0.0
count = 0
correct = 0
tokens = 0
with torch.no_grad():
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        total += loss.item()
        count += 1
        correct += (logits.argmax(dim=-1) == y).sum().item()
        tokens += y.numel()
        if count >= args.steps:
            break

loss = total / max(1, count)
print(f"checkpoint={args.checkpoint}")
print(f"device={device}")
print(f"eval_tokens={tokens:,}")
print(f"loss={loss:.4f}")
print(f"perplexity={math.exp(min(20.0, loss)):.2f}")
print(f"token_accuracy={correct / max(1, tokens):.4%}")
