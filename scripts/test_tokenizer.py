import argparse
import sys
from pathlib import Path
sys.path.insert(0, "src")
from llm.bpe import ByteBPE

p = argparse.ArgumentParser()
p.add_argument("--tokenizer", required=True)
p.add_argument("--text", default="Hello, world! こんにちは dunia 🌍")
a = p.parse_args()

tok = ByteBPE.load(a.tokenizer)
ids = tok.encode(a.text)
decoded = tok.decode(ids)
if decoded != a.text:
    raise AssertionError(f"round-trip failed: {decoded!r} != {a.text!r}")
print(f"tokenizer test: PASS tokens={len(ids)} vocab={tok.vocab_size}")
