# P2 prereg — probes on the cosine's task, and pre-flip signal beyond turn position — DRAFT

Authorized by: meta-review ask 2 ("align the evaluation tasks across cosine drift and probing");
reviewer paET's task-mismatch point. Locked before any nested-AUC probe result is computed.
Design reviews: pending.

## Motivation (frozen)
The cosine was evaluated on ever-flip (question level); the probes on pre-flip (turn level). The
pre-flip "nonlinear" result is reproduced by turn index alone (ERRATA E3), and the transfer negative
was degenerate (E4). Two arms: A puts probes on the cosine's task; B asks whether any pre-flip signal
survives turn position.

## Positive control (precondition for every probe result in a model)
Using the identical pipeline below on T0 states, the probe must decode the prompt condition (3-way,
macro one-vs-rest AUC, 5-fold, seed 20261021) at ≥ 0.90 at its best layer. A model that fails makes
its probe results "sub-threshold, not a null".

## Arm A — ever-flip task (same labels, questions and cells as the cosine)
- Features at layer ℓ (rows 1..L): `[h_T0, h_T1 − h_T0]`. Probe pipeline, all fit inside training
  folds: StandardScaler → PCA(64, random_state=20261021) → LogisticRegression(C=1, max_iter=2000).
  Secondary: the same with an RBF-SVM (C=1, gamma="scale").
- Nested evaluation: outer 5-fold stratified (seeds 0–4); an inner 4-fold CV picks the layer by inner
  mean AUC; the outer fold scores held-out questions; AUC over all out-of-fold predictions, averaged
  over the 5 outer seeds. The cosine gets the same nested treatment (inner picks the layer; lower
  cosine = flip, the sign fixed in advance).
- Test: 200 label permutations, each rerunning the full nested procedure with outer seed 0;
  p = (1 + #perm ≥ observed) / 201; Holm over the 12 cells.
- Descriptive: the paired probe − cosine AUC difference per cell (bootstrap over questions, 2,000).

## Arm B — pre-flip task, beyond turn position
- Primary horizon: next-response (the published §6.3 horizon). Secondary: one-turn-ahead.
- Models, question-grouped 5-fold (seed 20261021): (a) turn index alone (one-hot t), logistic;
  (b) turn index + PCA64(h_t) at the inner-selected layer, logistic. ΔAUC = AUC(b) − AUC(a), out-of-fold.
- Test: 200 permutations of labels **within turn strata** (keeps the censoring-induced turn structure,
  which is the E3 confound); Holm over 12 cells. Descriptive: within-turn AUCs for t = 1–3.

## Gates (LOCKED; per arm, primary instrument = linear probe)
Arm A:
- **[Sub-threshold]** iff the model fails the positive control.
- **[Detected in k cells]** iff Holm-adjusted p < 0.05 in k ≥ 1 cells. Earned: "A linear probe on
  T0/T1 states predicts eventual position change above chance in k of 12 cells (nested out-of-fold
  AUC = …)."
- **[Not detected]** iff k = 0 and the control passes. Earned: "We detect no linear-probe signal of
  eventual position change at this sample size (nested AUC X–Y; see C9 for power)." FORBIDDEN: "linear
  probes fail", "not linearly encoded", "nonlinear geometry".
Arm B:
- **[Sub-threshold]** as above.
- **[Beyond turn position in k cells]** iff Holm-adjusted p < 0.05 for ΔAUC > 0 in k ≥ 1 cells. Earned:
  "At the next-response horizon, hidden states add predictive information beyond turn position in k of
  12 cells (ΔAUC = …)."
- **[Not beyond turn position]** iff k = 0. Earned: "No pre-flip signal beyond turn position is
  detected at this n." FORBIDDEN: "pre-flip states are not separable", "nonlinear", "curved".

## Standing rules
No accuracy metric gates anything (class balance spans 66–91%); AUC only. The secondary instrument
and horizon are reported in full but cannot rescue a primary branch.
