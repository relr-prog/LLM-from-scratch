import json
from pathlib import Path
import torch
from torch.utils.data import Dataset

TEXT_EXTENSIONS = {".txt", ".md", ".json", ".jsonl"}

def extract_text(path):
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix == ".json":
        try:
            obj = json.loads(raw)
            if isinstance(obj, list):
                return "\n".join(str(x.get("text", x)) if isinstance(x, dict) else str(x) for x in obj)
            return str(obj.get("text", obj)) if isinstance(obj, dict) else str(obj)
        except json.JSONDecodeError:
            return raw
    if path.suffix == ".jsonl":
        out = []
        for line in raw.splitlines():
            try:
                obj = json.loads(line)
                out.append(str(obj.get("text", obj)) if isinstance(obj, dict) else str(obj))
            except json.JSONDecodeError:
                continue
        return "\n".join(out)
    return raw

def collect_text(root):
    parts = []
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and p.suffix.lower() in TEXT_EXTENSIONS:
            text = extract_text(p).strip()
            if text:
                parts.append(text)
    return "\n\n".join(parts)

class TokenDataset(Dataset):
    def __init__(self, path, block_size):
        self.data = torch.from_file(path, dtype=torch.int32)
        self.block_size = block_size

    def __len__(self):
        return max(0, self.data.numel() - self.block_size)

    def __getitem__(self, i):
        x = self.data[i:i+self.block_size].long()
        return x, self.data[i+1:i+self.block_size+1].long()
