# P0 prereg — which prompt did each model generate from? — DRAFT

Authorized by: ERRATA E10's opened question and P4's need for log-probabilities computed on the
prompt the model actually saw. Locked before any decode is run. Design reviews: pending.

## Motivation (frozen)
E10's gate found Llama's stored state on the `<|eot_id|>` that `models.py`'s tokenizer appends
(`add_eos_token=True`). If the generation pipeline tokenized the same way, Llama generated after an
end-of-turn marker. Generation was greedy (`do_sample=False`, temperature 0), so re-decoding is
deterministic up to numerics and decides it.

## Design
- Models: the four probed models (C2). Prompts: the first 5 questions of each model's
  `base_multiturn_metadata.json` × turns {0, 2} = 10 prompts per model, rebuilt from `input_messages`
  through `models.py`'s `apply_template`.
- Variants: **V_tok** = the tokenization that matched the stored states in E10 (`models.py`'s tokenizer
  settings, default special tokens); **V_std** = the same templated string tokenized with
  `add_special_tokens=False`. If V_tok and V_std give identical token ids for a model, that model is
  recorded as "identical prompts" and needs no decode.
- Decode: greedy, `max_new_tokens=48`, fp16 on one GPU (b200-devel), the pinned model revisions of E10.
- Measure: the exact-match character prefix between the decoded text (stripped) and the stored
  response (`assistant_response`, which is `output.strip()`), capped at its first 200 characters.

## Gates (LOCKED with this prereg; per model; m(V) = prompts with matching prefix ≥ 80 characters)
- **[V_tok confirmed]** iff m(V_tok) ≥ 8 and m(V_std) ≤ 2. Earned wording for Llama: "Llama-3.1-8B's
  responses were generated from a prompt ending in an end-of-turn token appended by its tokenizer
  configuration." FORBIDDEN: any statement that this changed Llama's behaviour — not tested here.
- **[V_std confirmed]** iff m(V_std) ≥ 8 and m(V_tok) ≤ 2. Earned: "generation used the standard
  prompt; for Llama the stored hidden state sits one token after the generation prompt."
- **[Both match]** iff both ≥ 8: the variants do not change the greedy continuation for this model;
  P4 may use either.
- **[Instrument failure]** every other region: no conclusion; investigate numerics (fp32, original
  transformers version) before P4 uses this model's log-probabilities.

## Preconditions
The 10 prompts per model are listed (by question string and turn) and committed before decoding.

## Consequences fixed in advance
V_tok for Llama ⇒ a FAILURE_TAXONOMY row and a disclosed limitation for every Llama behavioural
result; P4 computes Llama log-probabilities on V_tok. V_std ⇒ P4 uses V_std; the E10 note on Llama's
state token is added to the methods.
