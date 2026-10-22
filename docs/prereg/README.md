# Resubmission preregistrations (P0–P5) — shared conventions

Authorized by: the ARR August #1220 meta-review (five suggested revisions) and the author's
instruction of 2026-10-21 to address them rigorously before revising the paper. Each prereg is
DRAFT until its design reviews are filed and it is marked **LOCKED <date>**. Nothing outcome-relevant
for an arm is computed before that arm locks.

| ID | Meta-review ask | Question |
|---|---|---|
| P0 | precondition for P4 | Which prompt did each model actually generate from (Llama `<|eot_id|>`)? |
| P1 | 1. label reliability | Do Haiku's turn labels agree with a human on a random sample? |
| P2 | 2. align cosine and probe tasks | Do probes detect what the cosine detects, on the same task; is there pre-flip signal beyond turn position? |
| P3 | 3. verify T0 correctness | How often is T0 correct, and do the behavioural results hold on T0-correct conversations? |
| P4 | 4. surface-feature baselines | Does the hidden-state cosine add anything beyond length, hedging and log-probability? |
| P5 | 5. layer-wise evidence | What do the layer profiles of drift and AUC show across all 12 cells? |

## Conventions binding every arm
- **C1 Data.** Responses, metadata and tensors: HF dataset `sohampadianeu/pre-sycophancy-study-data`
  at revision `955b6da423654f36ec526232c38be6e950d6a84a`. Judge labels: `analysis_claude/*judgements*.csv`
  at git `1f4da07` (blobs: DeepSeek `4509968f3ccc`, Qwen2.5 `a15ec5cb18ec`, Llama `dbfaae5e3576`,
  Qwen3.5 `e188b652f0bb`, Gemma `59595c2088fe`). CREPE gold: `false-presuppositions-setting/data/`
  (`questions.txt`, `presuppositions.txt`, `corrections.txt`, line-aligned; `test.jsonl`).
- **C2 Cells.** Hidden-state arms: 4 models (DeepSeek-R1-Distill-Qwen-7B, Qwen2.5-7B-Instruct,
  Llama-3.1-8B-Instruct, Qwen3.5-9B) × 3 system-prompt conditions = 12 cells. Behavioural and label
  arms add Gemma-2-9B (15 cells). A test never pools cells.
- **C3 Units.** Ever-flip: the question within a cell. Turn-level: the turn, nested in its
  conversation — every turn-level interval is a conversation-cluster bootstrap.
- **C4 Labels.** Ever-flip = any Haiku `judgement` True in turns 1–5. Pre-flip tasks censor after the
  first flip and name their horizon: *next-response* (`hs[q][t]` → label of response t) or
  *one-turn-ahead* (`hs[q][t]` → label of response t+1). The DV is "position change" unless an arm
  earns otherwise (P3).
- **C5 Layers.** Sweeps use rows 1..L (L = depth 28/28/32/32); row 0 is constant and excluded.
- **C6 Randomness.** Every stochastic step uses a stated seed (default 20261021); every committed
  output must regenerate byte-identically from its producer and the C1 snapshot.
- **C7 Execution and routing.** New code goes to code-verifier before compute. Outputs go to
  `analysis_claude/` only through scripts. Results go to results-auditor (numbers) and
  confound-interrogator (meaning) before any record entry; citable numbers freeze in
  `docs/CITABLE_NUMBERS.md` against C1.
- **C8 Already-seen outcomes (disclosed, so no arm pretends to be blind to them).** Seen: per-layer
  cosine AUCs and the sweep-corrected permutation results; ever-flip rates; first-flip turn
  distributions; the 2026-09-16 exploratory response-text and question-only baselines (their
  definitions are not on this machine and are not reused); the turn-index result of ERRATA E3; an
  accuracy-based ever-flip RBF-SVM transfer arm (E4). Not seen: any human label, T0 correctness, any
  log-probability, the P0 decode, the nested-AUC probe results of P2, the P4 incremental tests.
- **C9 Power.** At n = 112–176 questions per cell, single-test power at a true AUC of 0.60 is
  0.42–0.70; under a layer-sweep correction it is roughly 0.12–0.32 for a one-layer effect and
  0.3–0.6 for a 4–8-layer band (results-auditor, 2026-10-21). Every null below is worded against this.
