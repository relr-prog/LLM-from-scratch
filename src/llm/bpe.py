from collections import Counter
from pathlib import Path
import json
import re

PATTERN = re.compile(r"\s+|\w+|[^\w\s]", re.UNICODE)

class ByteBPE:
    def __init__(self, vocab_size=8192):
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 for byte-level BPE")
        self.vocab_size = vocab_size
        self.merges = {}
        self.token_to_id = {}
        self.id_to_token = {}

    def train(self, text):
        words = PATTERN.findall(text)
        sequences = [list(w.encode("utf-8")) for w in words if w]
        vocab = {i: bytes([i]) for i in range(256)}
        while len(vocab) < self.vocab_size:
            counts = Counter()
            for seq in sequences:
                counts.update(zip(seq, seq[1:]))
            if not counts:
                break
            pair, _ = counts.most_common(1)[0]
            new_id = len(vocab)
            vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
            self.merges[pair] = new_id
            sequences = [self._merge(seq, pair, new_id) for seq in sequences]
        self.id_to_token = vocab
        self.token_to_id = {tok: i for i, tok in vocab.items()}
        self.vocab_size = len(vocab)

    @staticmethod
    def _merge(seq, pair, new_id):
        out, i = [], 0
        while i < len(seq):
            if i + 1 < len(seq) and (seq[i], seq[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        return out

    def encode(self, text):
        ids = []
        for word in PATTERN.findall(text):
            seq = list(word.encode("utf-8"))
            while True:
                candidates = [(rank, i, pair) for i, pair in enumerate(zip(seq, seq[1:])) if pair in self.merges for rank in [self.merges[pair]]]
                if not candidates:
                    break
                _, i, pair = min(candidates)
                seq = self._merge(seq, pair, self.merges[pair])
            ids.extend(seq)
        return ids

    def decode(self, ids):
        return b"".join(self.id_to_token[int(i)] for i in ids).decode("utf-8", errors="replace")

    def save(self, directory):
        p = Path(directory)
        p.mkdir(parents=True, exist_ok=True)
        (p / "merges.json").write_text(
            json.dumps({f"{a},{b}": v for (a, b), v in self.merges.items()}),
            encoding="utf-8",
        )
        (p / "vocab.json").write_text(
            json.dumps({str(k): v.hex() for k, v in self.id_to_token.items()}),
            encoding="utf-8",
        )
        (p / "meta.json").write_text(
            json.dumps({"type": "byte_bpe", "vocab_size": self.vocab_size}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory):
        p = Path(directory)
        meta_path = p / "meta.json"
        vocab = json.loads((p / "vocab.json").read_text(encoding="utf-8"))
        obj = cls(max(256, int(json.loads(meta_path.read_text())["vocab_size"]) if meta_path.exists() else len(vocab)))
        obj.id_to_token = {int(k): bytes.fromhex(v) for k, v in vocab.items()}
        obj.token_to_id = {v: k for k, v in obj.id_to_token.items()}
        merges = json.loads((p / "merges.json").read_text(encoding="utf-8"))
        obj.merges = {tuple(map(int, k.split(","))): int(v) for k, v in merges.items()}
        obj.vocab_size = len(obj.id_to_token)
        return obj
