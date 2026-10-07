import sys
import torch
sys.path.insert(0, "src")
from llm.bpe import ByteBPE
from llm.config import ModelConfig
from llm.model import GPT

cfg = ModelConfig(vocab_size=256, block_size=32, n_layer=2, n_head=4, n_embd=128)
model = GPT(cfg)
x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
logits, loss = model(x, x)
assert logits.shape == (2, cfg.block_size, cfg.vocab_size)
assert loss.ndim == 0 and torch.isfinite(loss)
loss.backward()
assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())

text = "Hello, world! こんにちは dunia 🌍"
tok = ByteBPE(vocab_size=256)
tok.train(text)
ids = tok.encode(text)
assert ids and tok.decode(ids) == text

print("smoke test: PASS")
