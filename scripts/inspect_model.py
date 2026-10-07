import argparse
import json
import sys

import torch

sys.path.insert(0, "src")
from llm.config import ModelConfig
from llm.model import GPT

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/tiny.json")
args = ap.parse_args()

raw = json.loads(open(args.config, encoding="utf-8").read())
cfg = ModelConfig(**raw["model"])
model = GPT(cfg)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
bytes_fp32 = params * 4
bytes_bf16 = params * 2
approx_training_flops_per_token = 6 * params

print(f"parameters={params:,}")
print(f"fp32_parameter_memory={bytes_fp32 / 1024**2:.1f} MiB")
print(f"bf16_parameter_memory={bytes_bf16 / 1024**2:.1f} MiB")
print(f"approx_training_flops_per_token={approx_training_flops_per_token:,.0f}")
print(f"layers={cfg.n_layer} heads={cfg.n_head} hidden={cfg.n_embd} context={cfg.block_size}")
print(f"gradient_checkpointing={cfg.gradient_checkpointing}")
