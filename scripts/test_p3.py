import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, "src")
from llm.config import ModelConfig
from llm.sft_data import SFTDataset

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    tokenizer_dir = root / "tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "merges.json").write_text("[]", encoding="utf-8")
    vocab = {str(i): [i] for i in range(256)}
    (tokenizer_dir / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (tokenizer_dir / "meta.json").write_text(json.dumps({"vocab_size": 256}), encoding="utf-8")
    data = root / "sft.jsonl"
    data.write_text(json.dumps({
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"}
        ]
    }) + "\n", encoding="utf-8")
    ds = SFTDataset(data, tokenizer_dir, 64)
    x, y = ds[0]
    assert x.shape == y.shape == (64,)
    assert (y != -100).any()
    assert (y == -100).any()

cfg = ModelConfig(vocab_size=256, block_size=32, n_layer=1, n_head=2, n_embd=64)
assert cfg.vocab_size == 256
print("p3 smoke: ok")
