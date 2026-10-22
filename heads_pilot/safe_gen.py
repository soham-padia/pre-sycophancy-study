"""Load a model in a configuration whose cached generation is verified against full recomputation.

Under transformers 5.12.1 / torch 2.12.1+cu130 on this cluster, bf16 `generate()` with a KV cache is
not exact for Qwen2.5-7B: with sdpa it degenerates ("the the") within a few dozen tokens; with eager
attention it drifts later and leaks turn-start tokens mid-sentence (20% of responses; the original
data has 1 in 2,796). fp32 with a cache is exact (teacher-forced agreement 1.000 over 450 tokens), as
is bf16 without a cache. Plain forward passes were always exact (the E10 recomputation).

So every job that generates text first proves, on two test prompts and ~400 new tokens each, that the
greedily generated tokens are the argmax of one teacher-forced full forward pass over the same
sequence, trying configurations in a fixed order and using the first that passes: fp32 + sdpa, then
bf16 + eager, bf16 + sdpa, and bf16 without a cache. If none passes it raises, so dependent jobs never
start. A 48-token check missed the eager drift; the length is what makes the check bite.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, GenerationConfig

CREPE = Path(__file__).resolve().parents[1] / "false-presuppositions-setting" / "data" / "test.jsonl"
VARIANTS = [("fp32-sdpa", torch.float32, "sdpa", True), ("bf16-eager", torch.bfloat16, "eager", True),
            ("bf16-sdpa", torch.bfloat16, "sdpa", True), ("bf16-nocache", torch.bfloat16, "sdpa", False)]
MIN_AGREEMENT = 0.99


def test_prompts(tok) -> list[list[int]]:
    """Two fixed prompts: a single-turn question, and a ~1,000-token multi-turn exchange with pushback."""
    rows = [json.loads(line) for _, line in zip(range(40), open(CREPE, encoding="utf-8"))]
    answer = "\n\n".join(r["comment"] for r in rows[1:12])[:4000]
    system = {"role": "system", "content": "You are a helpful assistant."}
    conversations = [[system, {"role": "user", "content": rows[20]["question"]}],
                     [system, {"role": "user", "content": rows[0]["question"]},
                      {"role": "assistant", "content": answer}, {"role": "user", "content": "Are you sure?"}]]
    out = []
    for messages in conversations:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        out.append(tok(text, add_special_tokens=False)["input_ids"])
    return out


@torch.no_grad()
def verify(model, tok, eos_ids, use_cache: bool, n_new: int = 400) -> dict:
    cfg = GenerationConfig(max_new_tokens=n_new, do_sample=False, repetition_penalty=1.0,
                           eos_token_id=sorted(eos_ids), pad_token_id=tok.pad_token_id or sorted(eos_ids)[0])
    per_prompt = []
    for ids in test_prompts(tok):
        x = torch.tensor([ids], device=model.device)
        out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), generation_config=cfg, use_cache=use_cache)
        new = out[0, len(ids):]
        if len(new) == 0:
            return {"passed": False, "reason": "no tokens generated"}
        # Logits at position p predict token p + 1, so the prefix up to the last new token predicts all of them.
        full_next = model(input_ids=out[:, :-1]).logits[0, len(ids) - 1:].argmax(-1)
        agree = (full_next == new).float().mean().item()
        per_prompt.append({"prompt_tokens": len(ids), "n_new": int(len(new)), "argmax_agreement": round(agree, 4)})
    worst = min(p["argmax_agreement"] for p in per_prompt)
    return {"passed": worst >= MIN_AGREEMENT, "worst_agreement": worst, "per_prompt": per_prompt}


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
