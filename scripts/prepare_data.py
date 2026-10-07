import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
from llm.text import iter_documents

p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--output", required=True)
p.add_argument("--manifest", default=None)
p.add_argument("--min-chars", type=int, default=32)
a = p.parse_args()

seen = set()
documents = []
duplicates = 0
for path, text in iter_documents(a.input):
    if len(text) < a.min_chars:
        continue
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if digest in seen:
        duplicates += 1
        continue
    seen.add(digest)
    documents.append((path, text))

Path(a.output).parent.mkdir(parents=True, exist_ok=True)
with open(a.output, "w", encoding="utf-8", newline="\n") as f:
    for _, text in documents:
        f.write(text)
        f.write("\n\n")

manifest = {
    "documents": len(documents),
    "duplicates_removed": duplicates,
    "characters": sum(len(text) for _, text in documents),
    "sources": [{"path": str(path), "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "characters": len(text)} for path, text in documents],
}
manifest_path = Path(a.manifest) if a.manifest else Path(a.output).with_suffix(".manifest.json")
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"documents={len(documents):,} duplicates_removed={duplicates:,} characters={manifest['characters']:,}")
