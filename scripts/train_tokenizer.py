import argparse
import sys
from pathlib import Path
sys.path.insert(0, "src")
from llm.bpe import ByteBPE

p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--output", required=True)
p.add_argument("--vocab-size", type=int, default=8192)
a = p.parse_args()
tok = ByteBPE(a.vocab_size)
tok.train(Path(a.input).read_text(encoding="utf-8"))
tok.save(a.output)
print(f"vocab={len(tok.id_to_token):,}")
