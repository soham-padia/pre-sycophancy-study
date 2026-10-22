"""Load a model in a configuration whose cached generation is verified against full recomputation.

Under transformers 5.12.1 on this cluster, `generate()` produced degenerate multi-turn text ("the the",
leaked "user" turns) for Qwen2.5-7B while plain forward passes over the same prompts were exact (the
E10 recomputation matched stored states). So every job that generates text first proves, on a long
multi-turn test prompt, that each greedily generated token is the argmax of a full, cache-free forward
pass over the same prefix, trying configurations in a fixed order and using the first that passes:
bf16 + sdpa, bf16 + eager, fp32 + sdpa (small models only), then no KV cache (slow, always exact).
If none passes, it raises and the job fails, so dependent jobs never start.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, GenerationConfig

CREPE = Path(__file__).resolve().parents[1] / "false-presuppositions-setting" / "data" / "test.jsonl"
VARIANTS = [("bf16-sdpa", torch.bfloat16, "sdpa", True), ("bf16-eager", torch.bfloat16, "eager", True),
            ("fp32-sdpa", torch.float32, "sdpa", True), ("bf16-nocache", torch.bfloat16, "sdpa", False)]


def test_prompt(tok) -> list[int]:
    """A fixed multi-turn prompt of roughly 1,000 tokens: question, long answer, pushback."""
    rows = [json.loads(line) for _, line in zip(range(40), open(CREPE, encoding="utf-8"))]
    answer = "\n\n".join(r["comment"] for r in rows[1:12])[:4000]
    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": rows[0]["question"]},
                {"role": "assistant", "content": answer},
                {"role": "user", "content": "Are you sure?"}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False)["input_ids"]


@torch.no_grad()
def verify(model, tok, eos_ids, use_cache: bool, n_new: int = 48) -> dict:
    ids = test_prompt(tok)
    x = torch.tensor([ids], device=model.device)
    cfg = GenerationConfig(max_new_tokens=n_new, do_sample=False, repetition_penalty=1.0,
                           eos_token_id=sorted(eos_ids), pad_token_id=tok.pad_token_id or sorted(eos_ids)[0])
    out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), generation_config=cfg, use_cache=use_cache)
    new = out[0, len(ids):]
    if len(new) == 0:
        return {"passed": False, "reason": "no tokens generated"}
    # Logits at position p predict token p + 1, so the prefix up to the last new token predicts all of them.
    full_next = model(input_ids=out[:, :-1]).logits[0, len(ids) - 1:].argmax(-1)
    agree = (full_next == new).float().mean().item()
    return {"passed": agree >= 0.97, "argmax_agreement": round(agree, 4), "n_new": int(len(new)),
            "prompt_tokens": len(ids), "sample": tok.decode(new, skip_special_tokens=True)[:160]}


def load_verified(name: str, tok, eos_ids, allow_fp32: bool = True, device_map="cuda"):
    """Return (model, record) for the first configuration whose generation passes `verify`."""
    tried = []
    for label, dtype, attn, use_cache in VARIANTS:
        if dtype is torch.float32 and not allow_fp32:
            continue
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, device_map=device_map,
                                                     attn_implementation=attn).eval()
        result = verify(model, tok, eos_ids, use_cache)
        tried.append({"variant": label, **result})
        print(f"[safe_gen] {label}: {result}", flush=True)
        if result["passed"]:
            model._safe_use_cache = use_cache
            return model, {"variant": label, "use_cache": use_cache, "tried": tried}
        del model
        torch.cuda.empty_cache()
    raise RuntimeError(f"no configuration produced verified generation: {tried}")
