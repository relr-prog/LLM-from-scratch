import argparse
import array
import sys
from pathlib import Path
sys.path.insert(0, "src")
from llm.bpe import ByteBPE

p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--tokenizer", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
tok = ByteBPE.load(a.tokenizer)
ids = tok.encode(Path(a.input).read_text(encoding="utf-8"))
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
with open(a.output, "wb") as f:
    array.array("i", ids).tofile(f)
print(f"tokens={len(ids):,}")
