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
- validation loss and perplexity
- temperature, top-k and top-p sampling
- CI smoke tests

## P0 → P1 pipeline

```text
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
binary dataset
    ↓
Transformer pretraining
    ↓
checkpoint
    ↓
autoregressive generation
```

## P3 — SFT and evaluation

SFT uses JSONL chat records and masks every non-assistant token with `-100`, so the loss is applied only to assistant responses.

Example training:

```bash
python scripts/train_sft.py --config configs/sft-tiny.json --data data/sft/example.jsonl --tokenizer data/tokenizer --checkpoint checkpoints/sft-latest.pt
```

The deterministic evaluation suite supports `exact`, `contains`, and `prefix` checks:

```bash
python scripts/eval_suite.py --checkpoint checkpoints/sft-latest.pt --tokenizer data/tokenizer --suite data/eval/suite.jsonl
```

Keep the evaluation set held out from SFT. Language-model quality is measured separately by `scripts/evaluate.py` using loss, perplexity, and token accuracy. Safety/refusal evaluation is intentionally a separate rubric-based benchmark rather than a fake deterministic score.

## Quick start

Python 3.11+ and PyTorch are required.

```bash
pip install -e .

python scripts/prepare_data.py \
  --input data/raw \
  --output data/clean.txt

python scripts/train_tokenizer.py \
  --input data/clean.txt \
  --output data/tokenizer

python scripts/test_tokenizer.py \
  --tokenizer data/tokenizer

python scripts/encode.py \
  --input data/clean.txt \
  --tokenizer data/tokenizer \
  --output data/train.bin

python scripts/train.py \
  --config configs/tiny.json \
  --data data/train.bin

# Multi-GPU DDP
torchrun --standalone --nproc_per_node=2 scripts/train.py \
  --config configs/base-125m.json \
  --data data/train.bin

python scripts/evaluate.py \
  --checkpoint checkpoints/latest.pt \
  --data data/train.bin

python scripts/generate.py \
  --checkpoint checkpoints/latest.pt \
  --tokenizer data/tokenizer \
  --prompt "The future of computing"
```

The included corpus is only a smoke-test corpus. A useful model needs a substantially larger, high-quality dataset.

## Project layout

```text
src/llm/
  bpe.py       tokenizer
  config.py    model/training configuration
  data.py      corpus + binary dataset
  model.py     Transformer
  text.py      normalization + document iteration

scripts/
  prepare_data.py
  train_tokenizer.py
  test_tokenizer.py
  encode.py
  train.py
  evaluate.py
  inspect_model.py
  profile_model.py
  train_sft.py
  evaluate.py
  eval_suite.py
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
- supervised fine-tuning with assistant-only loss masking
- deterministic generation evaluation suite

**P3 — post-training**
- supervised fine-tuning
- chat formatting
- deterministic evaluation suite
- held-out validation
- safety/refusal benchmark with an explicit rubric

The model is trained from random initialization. No pretrained model weights are used.
