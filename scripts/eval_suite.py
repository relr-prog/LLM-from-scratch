import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, "src")
from llm.config import ModelConfig
from llm.model import GPT
from llm.bpe import ByteBPE


def generate(model, tokenizer, prompt, device, tokens, temperature, top_k, top_p):
    ids = tokenizer.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    y = model.generate(x, tokens, temperature, top_k, top_p)
    return tokenizer.decode(y[0].tolist())[len(tokenizer.decode(ids)):]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--suite", default="data/eval/suite.jsonl")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.9)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = GPT(ModelConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tokenizer = ByteBPE.load(args.tokenizer)

    rows = [json.loads(x) for x in Path(args.suite).read_text(encoding="utf-8").splitlines() if x.strip()]
    results = []
    for row in rows:
        output = generate(model, tokenizer, row["prompt"], device, args.tokens, args.temperature, args.top_k, args.top_p)
        expected = row.get("expected")
        kind = row.get("type", "contains")
        if kind == "exact":
            passed = output.strip() == expected.strip()
        elif kind == "contains":
            passed = expected.lower() in output.lower()
        elif kind == "prefix":
            passed = output.lstrip().lower().startswith(expected.lower())
        else:
            raise ValueError(f"unknown evaluation type: {kind}")
        results.append({"id": row["id"], "type": kind, "passed": passed, "output": output})

    passed = sum(item["passed"] for item in results)
    print(f"cases={len(results)}")
    print(f"passed={passed}")
    print(f"accuracy={passed / max(1, len(results)):.2%}")
    for item in results:
        status = "PASS" if item["passed"] else "FAIL"
        print(f"[{status}] {item['id']}: {item['output'].strip()!r}")


if __name__ == "__main__":
    main()
