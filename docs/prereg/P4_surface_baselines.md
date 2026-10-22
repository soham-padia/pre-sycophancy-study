# P4 prereg — does the hidden-state cosine add anything beyond surface features? — DRAFT

Authorized by: meta-review ask 4 ("include surface-feature baselines (such as token length and log
probabilities)"); reviewer paET. Locked before any surface feature is computed. Design reviews: pending.

## Motivation (frozen)
The representational framing is licensed only if the T0→T1 hidden-state cosine predicts eventual
position change beyond what the visible responses and the question already predict. Exploratory
2026-09-16 runs (C8) suggested surface and question-only features match the cosine; their definitions
are not reused — the features below are fixed here.

## Features (question level, ever-flip task, 12 cells)
- S1 T0 response length, S2 T1 response length (tokens, the model's own tokenizer), S3 = S2 − S1.
- S4 hedge density of T0, S5 of T1: case-insensitive whole-phrase hits per 100 words over the fixed
  list {might, may, perhaps, possibly, possible, could be, it seems, i think, i believe, not entirely,
  not necessarily, generally, typically, often, likely, unlikely, uncertain, i'm not sure, it depends,
  i apologize, apologies, sorry}.
- S6 mean per-token log-probability of the T0 response, S7 of the T1 response, teacher-forced in fp16 on
  the prompt variant P0 confirmed for that model.
- Q question-only model: TF-IDF (unigrams + bigrams, min_df = 2) of the question text.

## Models and test
- Surface model: L2 logistic (C = 1) on standardized S1–S7. Question model: L2 logistic on Q. Cosine:
  T0→T1 cosine at a layer chosen by inner CV (as in P2). All out-of-fold, stratified 5-fold, seeds 0–4,
  AUC averaged over seeds.
- **Primary:** ΔAUC_incr = AUC(surface + cosine) − AUC(surface), per cell. p by 1,000 permutations of
  the cosine column across questions (re-running the inner layer choice each time), seed 20261021;
  Holm over 12 cells.
- Secondary (descriptive): the same increment over the question model; each feature's standalone AUC
  with bootstrap CIs; the T1-exclusion variant (drop conversations whose first flip is at T1, since T1
  hedging can be the flip itself).

## Preconditions
- P0 decided for the model; on P0 [Instrument failure], S6–S7 are dropped for that model and its
  surface model is reported as "without log-probabilities".
- Feature extraction is checked against 10 hand-verified responses per model before the test runs.

## Gates (LOCKED)
- **[Internal beyond surface in k cells]** iff Holm-adjusted p < 0.05 for ΔAUC_incr > 0 in k ≥ 1 cells.
  Earned: "In k of 12 cells, the T0→T1 hidden-state cosine adds predictive information beyond response
  length, hedging and log-probability (ΔAUC = X [CI])."
- **[Not distinguishable from surface]** iff k = 0. Earned: "At this sample size the cosine signal is not
  distinguishable from surface features of the responses." FORBIDDEN in this branch: "internal
  signature", "representational process", "before it becomes visible", "hidden-state precursor" as a
  finding.
- Descriptive rule in both branches: wherever the surface or question model's AUC is at or above the
  cosine's, say so in the same sentence as the cosine number.
