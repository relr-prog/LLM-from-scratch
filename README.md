# LLM-from-scratch

A from-scratch GPT-style language model training stack.

The goal is to build the complete pipeline ourselves:
- corpus ingestion and cleaning
- byte-level BPE tokenizer
- token dataset packing
- decoder-only Transformer
- causal self-attention + RoPE
- RMSNorm + SwiGLU
- AdamW + warmup/cosine decay
- mixed precision, gradient accumulation and clipping
- checkpoints and resume
- validation loss/perplexity
- autoregressive generation

This repository intentionally starts with a small model so the entire system can be trained and debugged before scaling.

## Quick start

Python 3.11+ and PyTorch are recommended.

```bash
pip install -e .
python scripts/prepare_data.py --input data/raw --output data/clean.txt
python scripts/train_tokenizer.py --input data/clean.txt --output data/tokenizer
python scripts/encode.py --input data/clean.txt --tokenizer data/tokenizer --output data/train.bin
python scripts/train.py --config configs/tiny.json
python scripts/generate.py --checkpoint checkpoints/latest.pt --tokenizer data/tokenizer --prompt "The future of computing"
```

Put plain text, Markdown, JSON, or JSONL files under `data/raw/`.

## First target

The default tiny configuration is deliberately small. Its purpose is correctness, not benchmark performance. Once loss decreases and generation works, increase depth/width/context and dataset size.

## Training objective

For tokens x_0 ... x_n, the model learns:

P(x_t | x_0 ... x_{t-1})

using next-token cross entropy with a causal attention mask.

## License

Project-specific licensing should be added before public distribution.
