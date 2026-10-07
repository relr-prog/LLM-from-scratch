# LLM-from-scratch

A language model built from first principles with PyTorch.

## Architecture

- byte-level BPE tokenizer trained locally
- Unicode normalization and exact-content deduplication
- packed binary token dataset
- decoder-only Transformer
- causal self-attention
- RoPE positional encoding
- RMSNorm
- SwiGLU MLP
- tied token embedding / LM head
- AdamW optimizer
- warmup + cosine learning-rate schedule
- gradient accumulation and clipping
- CUDA BF16/FP16 mixed precision
- checkpoint/resume with RNG state and checkpoint rotation
- memory-mapped token dataset
- DDP distributed training
- gradient checkpointing
- token throughput metrics
- standalone evaluation and model inspection
- supervised fine-tuning with assistant-only loss masking
- deterministic generation evaluation suite

## P0 → P2 pipeline

```
raw documents
    ↓
normalize + deduplicate
    ↓
clean corpus + manifest
    ↓
train byte-BPE
    ↓
token IDs
    ↓
Transformer pretraining
    ↓
checkpoint
    ↓
autoregressive generation
```

## P3 — SFT

SFT uses JSONL chat records and masks every non-assistant token with `-100`, so the loss is applied only to assistant responses.

Example training:

```bash
python scripts/train_sft.py --config configs/sft-tiny.json --data data/sft/example.jsonl --tokenizer data/tokenizer --checkpoint checkpoints/sft-latest.pt
```

The SFT checkpoint can be passed to the existing generation script.

## P3 — evaluation suite

The deterministic suite supports `exact`, `contains`, and `prefix` checks:

```bash
python scripts/eval_suite.py --checkpoint checkpoints/sft-latest.pt --tokenizer data/tokenizer --suite data/eval/suite.jsonl
```

Keep the evaluation set held out from SFT. `scripts/evaluate.py` separately reports language-model loss, perplexity, and token accuracy.

Safety/refusal evaluation is intentionally a separate rubric-based benchmark rather than a fake deterministic score.

## Quick start

Python 3.11+ and PyTorch are required.

```bash
pip install -e .

python scripts/prepare_data.py --input data/raw --output data/clean.txt
python scripts/train_tokenizer.py --input data/clean.txt --output data/tokenizer
python scripts/test_tokenizer.py --tokenizer data/tokenizer
python scripts/encode.py --input data/clean.txt --tokenizer data/tokenizer --output data/train.bin

python scripts/train.py --config configs/tiny.json --data data/train.bin

# Multi-GPU DDP
torchrun --standalone --nproc_per_node=2 scripts/train.py --config configs/base-125m.json --data data/train.bin
```

## Project layout

```
src/llm/
  bpe.py
  config.py
  data.py
  mmap_dataset.py
  model.py
  sft_data.py
  text.py

scripts/
  prepare_data.py
  train_tokenizer.py
  test_tokenizer.py
  encode.py
  train.py
  train_sft.py
  evaluate.py
  eval_suite.py
  inspect_model.py
  profile_model.py
  generate.py
  smoke_test.py
  test_p2.py
  test_p3.py

configs/
  tiny.json
  base-125m.json
  sft-tiny.json
```

## Scaling roadmap

**P0 — correctness**
- tiny model
- tokenizer
- forward/backward
- generation
- CI

**P1 — data and training quality**
- normalization
- deduplication
- manifests
- reproducible tokenizer artifacts
- numerical stability checks

**P2 — scale**
- memory-mapped datasets
- larger context
- larger model configurations
- checkpoint rotation + RNG state
- DDP distributed training
- gradient checkpointing
- throughput profiling
- standalone evaluation

**P3 — post-training**
- supervised fine-tuning
- chat formatting
- deterministic evaluation suite
- held-out validation
- safety/refusal benchmark with an explicit rubric

The model is trained from random initialization. No pretrained model weights are used.
