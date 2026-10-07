import argparse
import json
import sys
import time
import torch
sys.path.insert(0, "src")
from llm.config import ModelConfig
from llm.model import GPT

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/tiny.json")
p.add_argument("--steps", type=int, default=10)
a = p.parse_args()

cfg = ModelConfig(**json.loads(open(a.config, encoding="utf-8").read())["model"])
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT(cfg).to(device)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
x = torch.randint(0, cfg.vocab_size, (1, cfg.block_size), device=device)

for _ in range(2):
    _, loss = model(x, x)
    loss.backward()
    model.zero_grad(set_to_none=True)

if device == "cuda":
    torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(a.steps):
    _, loss = model(x, x)
    loss.backward()
    model.zero_grad(set_to_none=True)
if device == "cuda":
    torch.cuda.synchronize()
elapsed = time.perf_counter() - start
tokens = a.steps * x.numel()
print(f"device={device}")
print(f"parameters={params:,}")
print(f"tokens_per_sec={tokens / elapsed:,.0f}")
print(f"step_ms={elapsed / a.steps * 1000:.2f}")
