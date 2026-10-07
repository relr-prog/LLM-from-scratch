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
p.add_argument("--dtype", choices=["int32", "int64"], default="int32")
a = p.parse_args()

tok = ByteBPE.load(a.tokenizer)
text = Path(a.input).read_text(encoding="utf-8")
ids = tok.encode(text)
if not ids:
    raise ValueError("Input produced zero tokens.")

Path(a.output).parent.mkdir(parents=True, exist_ok=True)
typecode = "i" if a.dtype == "int32" else "q"
with open(a.output, "wb") as f:
    array.array(typecode, ids).tofile(f)
print(f"tokens={len(ids):,} dtype={a.dtype} bytes={len(ids) * (4 if typecode == 'i' else 8):,}")
