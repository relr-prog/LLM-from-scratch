import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "src")
from llm.config import ModelConfig, TrainConfig
from llm.data import TokenDataset
from llm.model import GPT

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def lr_at(step, cfg):
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / max(1, cfg.warmup_steps)
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps))
    return cfg.min_learning_rate + 0.5 * (1.0 + math.cos(math.pi * progress)) * (cfg.learning_rate - cfg.min_learning_rate)

@torch.no_grad()
def evaluate(model, loader, device, steps):
    model.eval()
    total = 0.0
    count = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total += loss.item()
        count += 1
        if count >= steps:
            break
    model.train()
    return total / max(1, count)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tiny.json")
    ap.add_argument("--data", default="data/train.bin")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg_json = json.loads(Path(args.config).read_text(encoding="utf-8"))
    mcfg = ModelConfig(**cfg_json["model"])
    tcfg = TrainConfig(**cfg_json["train"])
    set_seed(tcfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16

    tokens = torch.from_file(args.data, shared=False, size=-1, dtype=torch.int32)
    n = tokens.numel()
    split = int(n * 0.9)
    del tokens
    if split <= mcfg.block_size or n - split <= mcfg.block_size:
        raise ValueError("Dataset is too small. Add more training text.")
    train_ds = TokenDataset(args.data, mcfg.block_size, 0, split)
    val_ds = TokenDataset(args.data, mcfg.block_size, split, n)
    train_loader = DataLoader(train_ds, batch_size=tcfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=tcfg.batch_size, shuffle=False, drop_last=True)

    model = GPT(mcfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.learning_rate, betas=(0.9, 0.95), weight_decay=tcfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1

    Path(tcfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    print(f"device={device} parameters={sum(p.numel() for p in model.parameters()):,} tokens={n:,}")

    iterator = iter(train_loader)
    for step in range(start_step, tcfg.max_steps):
        lr = lr_at(step, tcfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0

        for _ in range(tcfg.grad_accum_steps):
            try:
                x, y = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                x, y = next(iterator)
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                _, loss = model(x, y)
                loss = loss / tcfg.grad_accum_steps
            loss_value += loss.item()
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step % 25 == 0:
            print(f"step={step:6d} loss={loss_value:.4f} lr={lr:.3e}")

        if step % tcfg.eval_interval == 0 or step == tcfg.max_steps - 1:
            val_loss = evaluate(model, val_loader, device, tcfg.eval_steps)
            print(f"eval step={step:6d} val_loss={val_loss:.4f} ppl={math.exp(min(20.0, val_loss)):.2f}")
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_config": mcfg.__dict__,
                "train_config": tcfg.__dict__,
            }, Path(tcfg.checkpoint_dir) / "latest.pt")

if __name__ == "__main__":
    main()
