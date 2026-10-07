import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, "src")
from llm.config import ModelConfig, TrainConfig
from llm.mmap_dataset import MMapTokenDataset
from llm.model import GPT


def distributed_state(mode):
    requested = mode
    active = requested == "ddp" or (
        requested == "auto" and int(os.environ.get("WORLD_SIZE", "1")) > 1
    )
    if not active:
        return False, 0, 1, 0
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    return True, rank, world_size, local_rank


def set_seed(seed, rank=0):
    value = seed + rank
    random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def lr_at(step, cfg):
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / max(1, cfg.warmup_steps)
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps))
    return cfg.min_learning_rate + 0.5 * (1.0 + math.cos(math.pi * progress)) * (cfg.learning_rate - cfg.min_learning_rate)


def reduce_mean(value, device, enabled):
    tensor = torch.tensor(float(value), device=device)
    if enabled:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return tensor.item()


@torch.no_grad()
def evaluate(model, loader, device, steps, distributed):
    model.eval()
    total = 0.0
    count = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        _, loss = model(x, y)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite validation loss")
        total += loss.item()
        count += 1
        if count >= steps:
            break
    model.train()
    return reduce_mean(total / max(1, count), device, distributed)


def rng_state():
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def collect_rng_states(distributed):
    local = rng_state()
    if not distributed:
        return [local]
    states = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(states, local)
    return states


def restore_rng_state(state):
    if not state:
        return
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def rotate_checkpoints(directory, keep):
    paths = sorted(Path(directory).glob("step_*.pt"))
    for path in paths[:-keep]:
        path.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tiny.json")
    ap.add_argument("--data", default="data/train.bin")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg_json = json.loads(Path(args.config).read_text(encoding="utf-8"))
    mcfg = ModelConfig(**cfg_json["model"])
    tcfg = TrainConfig(**cfg_json["train"])
    distributed, rank, world_size, local_rank = distributed_state(tcfg.distributed)

    if torch.cuda.is_available():
        if distributed:
            torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank if distributed else 0)
    else:
        device = torch.device("cpu")

    set_seed(tcfg.seed, rank)
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16

    data_size = Path(args.data).stat().st_size // 4
    split = int(data_size * 0.9)
    if split <= mcfg.block_size or data_size - split <= mcfg.block_size:
        raise ValueError("Dataset is too small. Add more training text.")

    train_ds = MMapTokenDataset(args.data, mcfg.block_size, 0, split)
    val_ds = MMapTokenDataset(args.data, mcfg.block_size, split, data_size)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None
    train_loader = DataLoader(train_ds, batch_size=tcfg.batch_size, sampler=train_sampler, shuffle=train_sampler is None, drop_last=True, pin_memory=use_amp)
    val_loader = DataLoader(val_ds, batch_size=tcfg.batch_size, sampler=val_sampler, shuffle=False, drop_last=True, pin_memory=use_amp)

    model = GPT(mcfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tcfg.learning_rate, betas=(0.9, 0.95), weight_decay=tcfg.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt["step"] + 1
        states = ckpt.get("rng_states")
        if states:
            restore_rng_state(states[min(rank, len(states) - 1)])

    if distributed:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    raw_model = model.module if isinstance(model, DDP) else model
    checkpoint_dir = Path(tcfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        print(
            f"device={device} world_size={world_size} parameters={count_parameters(raw_model):,} "
            f"tokens={data_size:,} gradient_checkpointing={mcfg.gradient_checkpointing}"
        )

    iterator = iter(train_loader)
    epoch = 0
    step_start = time.perf_counter()

    for step in range(start_step, tcfg.max_steps):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        lr = lr_at(step, tcfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0

        for _ in range(tcfg.grad_accum_steps):
            try:
                x, y = next(iterator)
            except StopIteration:
                epoch += 1
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                iterator = iter(train_loader)
                x, y = next(iterator)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                _, loss = model(x, y)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite training loss at step {step}")
                loss = loss / tcfg.grad_accum_steps
            loss_value += loss.item()
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gradient at step {step}")
        scaler.step(optimizer)
        scaler.update()

        if step % 25 == 0 or step == tcfg.max_steps - 1:
            elapsed = max(time.perf_counter() - step_start, 1e-6)
            tokens_per_sec = ((step - start_step + 1) * tcfg.batch_size * tcfg.grad_accum_steps * mcfg.block_size * world_size) / elapsed
            if rank == 0:
                print(
                    f"step={step:6d} loss={loss_value:.4f} grad={float(grad_norm):.3f} "
                    f"lr={lr:.3e} tokens/s={tokens_per_sec:,.0f}"
                )

        if step % tcfg.eval_interval == 0 or step == tcfg.max_steps - 1:
            val_loss = evaluate(model, val_loader, device, tcfg.eval_steps, distributed)
            rng_states = collect_rng_states(distributed)
            if rank == 0:
                payload = {
                    "step": step,
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "model_config": mcfg.__dict__,
                    "train_config": tcfg.__dict__,
                    "rng_states": rng_states,
                }
                step_path = checkpoint_dir / f"step_{step:08d}.pt"
                save_checkpoint(step_path, payload)
                save_checkpoint(checkpoint_dir / "latest.pt", payload)
                rotate_checkpoints(checkpoint_dir, tcfg.keep_checkpoints)
                print(f"eval step={step:6d} val_loss={val_loss:.4f} ppl={math.exp(min(20.0, val_loss)):.2f}")
            if distributed:
                dist.barrier()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
