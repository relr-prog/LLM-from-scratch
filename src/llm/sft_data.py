import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .bpe import ByteBPE


def render_messages(messages):
    parts = []
    for message in messages:
        role = message["role"].strip().lower()
        content = message["content"].strip()
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported role: {role}")
        parts.append(f"<|{role}|>\n{content}\n")
    parts.append("<|assistant|>\n")
    return "".join(parts)


class SFTDataset(Dataset):
    """JSONL chat dataset with labels masked outside assistant responses."""

    def __init__(self, path, tokenizer_path, block_size):
        self.tokenizer = ByteBPE.load(tokenizer_path)
        self.block_size = block_size
        self.examples = []
        for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            item = json.loads(line)
            messages = item.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"line {line_no}: messages must be a non-empty list")
            self.examples.append(self._encode(messages))

    def _encode(self, messages):
        ids = []
        labels = []
        for message in messages:
            role = message["role"].strip().lower()
            content = message["content"].strip()
            prefix = self.tokenizer.encode(f"<|{role}|>\n")
            body = self.tokenizer.encode(content + "\n")
            ids.extend(prefix)
            labels.extend([-100] * len(prefix))
            ids.extend(body)
            labels.extend(body if role == "assistant" else [-100] * len(body))
        if not messages or messages[-1]["role"].lower() != "assistant":
            prompt = self.tokenizer.encode("<|assistant|>\n")
            ids.extend(prompt)
            labels.extend([-100] * len(prompt))
        ids = ids[: self.block_size]
        labels = labels[: self.block_size]
        if not any(label != -100 for label in labels):
            raise ValueError("SFT example has no assistant tokens inside block_size")
        return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        ids, labels = self.examples[index]
        x = torch.zeros(self.block_size, dtype=torch.long)
        y = torch.full((self.block_size,), -100, dtype=torch.long)
        length = min(ids.numel(), self.block_size)
        x[:length] = ids[:length]
        y[:length] = labels[:length]
        return x, y
