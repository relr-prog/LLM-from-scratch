import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, "src")
from llm.config import ModelConfig, TrainConfig
from llm.model import GPT
from llm.sft_data import SFTDataset


def lr_at(step, cfg):
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / max(1, cfg.warmup_steps)
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps))
    return cfg.min_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (cfg.learning_rate - cfg.min_learning_rate)


def evaluate(model, loader, device, steps):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            total += loss.item()
            count += 1
            if count >= steps:
                break
    model.train()
    return total / max(1, count)


ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/sft-tiny.json")
ap.add_argument("--data", default="data/sft/example.jsonl")
ap.add_argument("--tokenizer", default="data/tokenizer")
ap.add_argument("--checkpoint", default="checkpoints/sft-latest.pt")
args = ap.parse_args()

raw = json.loads(Path(args.config).read_text(encoding="utf-8"))
mcfg = ModelConfig(**raw["model"])
tcfg = TrainConfig(**raw["train"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(tcfg.seed)
torch.manual_seed(tcfg.seed)

dataset = SFTDataset(args.data, args.tokenizer, mcfg.block_size)
if len(dataset) < 2:
    raise ValueError("SFT dataset needs at least 2 examples")
n_val = max(1, len(dataset) // 10)
n_train = len(dataset) - n_val
train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(tcfg.seed))
train_loader = DataLoader(train_ds, batch_size=tcfg.batch_size, shuffle=True, drop_last=False)
val_loader = DataLoader(val_ds, batch_size=tcfg.batch_size, shuffle=False, drop_last=False)

model = GPT(mcfg).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.learning_rate, betas=(0.9, 0.95), weight_decay=tcfg.weight_decay)
use_amp = device.type == "cuda"
amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

best = float("inf")
iterator = iter(train_loader)
for step in range(tcfg.max_steps):
    lr = lr_at(step, tcfg)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    try:
        x, y = next(iterator)
    except StopIteration:
        iterator = iter(train_loader)
        x, y = next(iterator)
    x, y = x.to(device), y.to(device)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        _, loss = model(x, y)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite SFT loss at step {step}")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
    scaler.step(optimizer)
    scaler.update()

    if step % tcfg.eval_interval == 0 or step == tcfg.max_steps - 1:
        val_loss = evaluate(model, val_loader, device, tcfg.eval_steps)
        print(f"step={step:5d} train_loss={loss.item():.4f} val_loss={val_loss:.4f} ppl={math.exp(min(20, val_loss)):.2f}")
        if val_loss < best:
            best = val_loss
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "model_config": mcfg.__dict__,
                "train_config": tcfg.__dict__,
                "best_val_loss": best,
            }, args.checkpoint)
