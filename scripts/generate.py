import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, "src")
from llm.bpe import ByteBPE
from llm.config import ModelConfig
from llm.model import GPT

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/latest.pt")
    ap.add_argument("--tokenizer", default="data/tokenizer")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = GPT(ModelConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tokenizer = ByteBPE.load(args.tokenizer)
    ids = tokenizer.encode(args.prompt)
    if not ids:
        raise ValueError("Prompt produced zero tokens.")
    x = torch.tensor([ids], dtype=torch.long, device=device)
    y = model.generate(x, args.tokens, args.temperature, args.top_k, args.top_p)
    print(tokenizer.decode(y[0].tolist()))

if __name__ == "__main__":
    main()
