#!/usr/bin/env python3
"""Stage 4 of the heads pilot: regenerate conversations with chosen attention heads scaled.

Steering half = odd qids (never used for ranking). A head is scaled by multiplying its slice of the
`o_proj` input by alpha at every position of every forward pass while steering is on. By default
steering starts at the first pressure turn, so T0 is the unsteered stage-1 response and any effect
lands on the turns that respond to the user's follow-ups. `--from-turn 0 --turns 1` instead steers a
single T0 answer (side-effect check: does the model still reject false presuppositions?).

Conditions (array task id = condition * n_shards + shard):
  baseline (alpha 1; must reproduce stage 1 exactly), sel_off / sel_up (top-k heads by flip score,
  alpha 0 / 2), rand_off / rand_up (layer-matched random heads, alpha 0 / 2).

Batch size 1. Writes `<gen>/steer/k<k>[_t0]/<condition>/<arm>.shard{i}of{n}.jsonl`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, GenerationConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import LADDERS, SYSTEM_PROMPTS, generate_batch, render, strip_reasoning  # noqa: E402
from safe_gen import load_verified  # noqa: E402

CONDITIONS = [("baseline", "selected", 1.0), ("sel_off", "selected", 0.0), ("sel_up", "selected", 2.0),
              ("rand_off", "random", 0.0), ("rand_up", "random", 2.0)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--condition-set", default="base")
    ap.add_argument("--gen-dir", type=Path,
                    default=Path("/work/neu/p2026_0037_neu/heads_pilot/Qwen2.5-7B-Instruct/base"))
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--task", type=int, required=True, help="array task id")
    ap.add_argument("--n-shards", type=int, default=8)
    ap.add_argument("--arms", nargs="+", default=["disagree", "agree"])
    ap.add_argument("--from-turn", type=int, default=1)
    ap.add_argument("--turns", type=int, default=6)
    args = ap.parse_args()
    name, head_set, alpha = CONDITIONS[args.task // args.n_shards]
    shard = args.task % args.n_shards

    ranking = json.loads((args.gen_dir / "heads" / "ranking.json").read_text())
    heads = [tuple(x) for x in ranking["selected" if head_set == "selected" else "random_control"][str(args.k)]]

    tok = AutoTokenizer.from_pretrained(args.model)
    eos = GenerationConfig.from_pretrained(args.model).eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    model, safe = load_verified(args.model, tok, eos_ids)          # verified before any hook is added
    n_heads = model.config.num_attention_heads
    head_dim = getattr(model.config, "head_dim", None) or model.config.hidden_size // n_heads
    # Decode exactly as stage 1 did (read from its run_meta files), or baseline cannot reproduce it.
    decoding = {(m["generation"]["repetition_penalty"], m["generation"]["max_new_tokens"])
                for m in (json.loads(p.read_text()) for p in sorted(args.gen_dir.glob("run_meta*.json")))}
    assert len(decoding) == 1, f"stage-1 run_meta files missing or disagreeing on decoding: {decoding}"
    rep_penalty, max_new_tokens = decoding.pop()
    gen_cfg = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=rep_penalty,
                               eos_token_id=sorted(eos_ids), pad_token_id=tok.pad_token_id)
    print(f"stage-1 decoding: repetition_penalty={rep_penalty}, max_new_tokens={max_new_tokens}", flush=True)

    state = {"on": False}
    by_layer: dict[int, list[int]] = {}
    for l, h in heads:
        by_layer.setdefault(l, []).append(h)

    def hook(layer_heads):
        def pre(_module, inputs):
            if not state["on"] or alpha == 1.0:
                return None
            x = inputs[0].clone()
            for h in layer_heads:
                x[..., h * head_dim:(h + 1) * head_dim] *= alpha
            return (x,) + tuple(inputs[1:])
        return pre

    handles = [model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(hook(hs))
               for l, hs in by_layer.items()]

    tag = f"k{args.k}" + ("" if args.from_turn >= 1 else "_t0")
    out_dir = args.gen_dir / "steer" / tag / name
    out_dir.mkdir(parents=True, exist_ok=True)
    system = {"role": "system", "content": SYSTEM_PROMPTS[args.condition_set]}
    for arm in args.arms:
        base = sorted((json.loads(line) for f in sorted(args.gen_dir.glob(f"{arm}.shard*.jsonl"))
                       for line in open(f, encoding="utf-8")), key=lambda c: c["qid"])
        base = [c for c in base if c["qid"] % 2 == 1]
        base = [c for i, c in enumerate(base) if i % args.n_shards == shard]
        with open(out_dir / f"{arm}.shard{shard}of{args.n_shards}.jsonl", "w", encoding="utf-8") as f:
            for c in base:
                history, turns = [system], []
                for t in range(args.turns):
                    user = c["question"] if t == 0 else LADDERS[arm][t - 1]
                    if t < args.from_turn:
                        r = {k: c["turns"][t][k] for k in ("response", "n_tokens", "finished")}
                        r["text"] = r.pop("response")
                    else:
                        state["on"] = True
                        r = generate_batch(model, tok, [render(tok, history + [{"role": "user", "content": user}])],
                                           gen_cfg, eos_ids)[0]
                        state["on"] = False
                    turns.append({"t": t, "user": user, "response": r["text"], "n_tokens": r["n_tokens"],
                                  "finished": r["finished"], "steered": t >= args.from_turn})
                    history += [{"role": "user", "content": user},
                                {"role": "assistant", "content": strip_reasoning(r["text"])}]
                f.write(json.dumps({"qid": c["qid"], "question": c["question"], "arm": arm, "condition": name,
                                    "heads": heads, "alpha": alpha, "gen_config": safe["variant"], "turns": turns},
                                   ensure_ascii=False) + "\n")
                f.flush()
    for h in handles:
        h.remove()
    print(f"{name} shard {shard}: done", flush=True)


if __name__ == "__main__":
    main()
