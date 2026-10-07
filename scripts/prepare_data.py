import argparse
import sys
from pathlib import Path
sys.path.insert(0, "src")
from llm.data import collect_text

p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
text = collect_text(a.input)
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
Path(a.output).write_text(text, encoding="utf-8")
print(f"wrote {len(text):,} characters")
