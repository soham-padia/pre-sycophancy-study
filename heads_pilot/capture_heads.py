#!/usr/bin/env python3
"""Stage 3a of the heads pilot: what every attention head writes just before each response.

For each conversation and turn t (0-5), rebuilds the exact prompt the generator used (same template,
tokenization and history handling, imported from `generate.py`), runs one forward pass, and keeps each
head's attention output at the last prompt token: the input to `o_proj`, split per head, i.e. the
vector the head contributes before the output projection. Pre-response, like the original study's
hidden states (ERRATA E10).

Batch size 1 (no padding). Writes `<gen>/heads/<arm>.pt` = {"qids": [...], "z": float16 tensor
[n_conv, 6, n_layers, n_heads, head_dim], "model": ..., "revision": ...}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import SYSTEM_PROMPTS, render, strip_reasoning  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--condition", default="base")
    ap.add_argument("--gen-dir", type=Path,
                    default=Path("/work/neu/p2026_0037_neu/heads_pilot/Qwen2.5-7B-Instruct/base"))
    ap.add_argument("--arms", nargs="+", default=["disagree", "agree"])
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    cfg = model.config
    n_layers, n_heads = cfg.num_hidden_layers, cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_heads
    captured = [None] * n_layers

    def hook(layer):
        def pre(_module, inputs):
            captured[layer] = inputs[0][0, -1, :].detach().float().cpu()
        return pre

    handles = [model.model.layers[i].self_attn.o_proj.register_forward_pre_hook(hook(i)) for i in range(n_layers)]
    out_dir = args.gen_dir / "heads"
    out_dir.mkdir(exist_ok=True)
    system = {"role": "system", "content": SYSTEM_PROMPTS[args.condition]}
    for arm in args.arms:
        convs = sorted((json.loads(line) for f in sorted(args.gen_dir.glob(f"{arm}*.jsonl"))
                        for line in open(f, encoding="utf-8")), key=lambda c: c["qid"])
        z = torch.zeros(len(convs), 6, n_layers, n_heads, head_dim, dtype=torch.float16)
        for ci, c in enumerate(convs):
            history = [system]
            for t, turn in enumerate(c["turns"]):
                ids = render(tok, history + [{"role": "user", "content": turn["user"]}])
                assert len(ids) == turn["prompt_tokens"], (c["qid"], t, len(ids), turn["prompt_tokens"])
                with torch.no_grad():
                    model(input_ids=torch.tensor([ids], device=model.device))
                z[ci, t] = torch.stack(captured).view(n_layers, n_heads, head_dim).half()
                history += [{"role": "user", "content": turn["user"]},
                            {"role": "assistant", "content": strip_reasoning(turn["response"])}]
        torch.save({"qids": [c["qid"] for c in convs], "z": z, "model": args.model,
                    "revision": getattr(cfg, "_commit_hash", None), "arm": arm}, out_dir / f"{arm}.pt")
        print(f"{arm}: {len(convs)} conversations, z {tuple(z.shape)}", flush=True)
    for h in handles:
        h.remove()


if __name__ == "__main__":
    main()
