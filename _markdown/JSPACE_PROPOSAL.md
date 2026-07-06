# Proposal: Is the Pre-Capitulation Signal Inside the Verbalizable Workspace?

**Status:** research proposal (follow-up to the EACL sycophancy paper)
**Data required:** existing hidden states + one new GPU computation (J-lens matrices)
**References:** Anthropic, "Verbalizable Representations Form a Global Workspace in Language Models," Transformer Circuits, 2026 (https://transformer-circuits.pub/2026/workspace/index.html); overview at https://www.anthropic.com/research/global-workspace

---

## 1. Background

Our EACL paper shows that sycophantic capitulation has a representational history:
cosine drift between T0 and T1 hidden states predicts eventual capitulation
(peak AUC 0.596–0.636, robust to T1-flip exclusion), but the signal is weak,
model-specific, layer-localized, and **not linearly decodable** — linear probes
sit at the majority baseline at every layer, while nonlinear classifiers recover
only part of the structure. We currently have no account of *what* the signal is.

Anthropic's global workspace work provides a candidate decomposition. The
**Jacobian lens (J-lens)** computes, per layer, the average linearized effect of
that layer's activations on final-layer output:

    J_l = E[ ∂h_final,t' / ∂h_l,t ]

averaged over token positions and ~1,000 prompts, composed with the unembedding.
It reveals which *verbalizable token concepts* a hidden state carries. The
**J-space** — sparse nonnegative combinations of J-lens vectors (k ≤ 25) —
behaves like a global workspace: middle layers (~38–92% relative depth)
broadcast abstract, persistent representations that disproportionately many
components read and write; routine processing bypasses it; and unverbalized
strategic reasoning ("leverage," "panic," "fake") surfaces in it before output.

## 2. Core Question

**Does the pre-capitulation signal live inside the verbalizable workspace
(J-space) or outside it?**

Both outcomes are findings:

| Outcome | Interpretation | Consequence |
|---|---|---|
| Signal concentrated in J-space | Capitulation transits the "deliberate" pathway; the model is, in a functional sense, already entertaining concession | Token-level readable early warning; auditable |
| Signal concentrated in the complement | The pre-capitulation state is genuinely **pre-verbal** — drift precedes workspace loading | Stronger version of "before it becomes visible"; explains why the signal is hard to decode |
| Split / layer-dependent | Workspace loading happens between T1 and the flip turn | Gives the *trajectory* claim a mechanistic timeline |

A secondary hypothesis: the sparse, overcomplete structure of J-space is exactly
the kind of geometry that defeats linear probes on raw activations while
remaining partially recoverable by nonlinear classifiers — a candidate
explanation for our central geometric finding.

## 3. Experiments

### E1. J-space decomposition of the cosine signal
1. Compute J-lens matrices for **Qwen2.5-7B-Instruct** (cleanest signal, most
   resistant model) over ~1,000 pretraining-like prompts on the V100 cluster.
   Use vector-Jacobian products accumulated per layer rather than full Jacobians.
2. Project the *already saved* T0/T1 hidden states onto J-space (sparse
   nonnegative decomposition, k ≤ 25) and its orthogonal complement.
3. Recompute the T0→T1 cosine-disruption AUC separately in each subspace,
   with the same bootstrap-CI and T1-exclusion protocol as the paper.

**Readout:** which subspace carries the AUC.

### E2. Token-level readout of pre-flip states
Apply the J-lens to T1 hidden states and rank token concepts. Test whether
concession/social-agreement concepts ("sorry," "right," "mistake," "agree,"
"user") load higher for eventual-flippers than holds — *before* any behavioral
flip (T1-flip cases excluded). This is the direct analogue of Anthropic's
alignment-auditing result, and would convert our opaque AUC into a
human-readable early warning.

### E3. Layer-band mapping
Map our per-model peak-AUC layers onto the sensory/workspace/motor tripartition
(estimated per model via the paper's diagnostics: J-lens content, kurtosis,
autocorrelation, top-token prediction rise). Prediction: peaks inside the
workspace band are more robust to T1-exclusion than peaks in the motor band
(Llama's layer-32 peak is the test case — it sits at 97% depth).

### E4 (stretch). Causal test
Using NNsight, attenuate the J-space component of the T0→T1 drift at inference
time and measure the change in flip rate over T2–T5. Requires generation with
intervention on the cluster; only attempt if E1 shows workspace concentration.

## 4. Feasibility

- **Data:** all hidden states already extracted (this is why we saved them per-layer).
- **New compute:** J-lens matrices per model — backprop through a 7–9B model,
  ~1,000 prompts × per-layer VJP accumulation. Fits on one V100-32GB in fp16
  with activation checkpointing; estimated a few 8-hour job cycles per model.
- **Reference implementations** exist for open-weight models (Neuronpedia).
- **Start with one model** (Qwen2.5-7B); add Llama-3.1-8B for the motor-band
  contrast (E3) if time allows.

## 5. Risks and caveats

1. The workspace findings are from Claude-scale models; 7–9B models may have a
   less crisp workspace region. (Mitigation: the paper validated on Haiku-class
   models; E3's band-estimation diagnostics tell us early whether structure exists.)
2. Vocabulary-alignment limits: number-words swapped poorly in the original
   paper; concession concepts are common tokens, so this risk is low for E2.
3. J-lens averages over prompt distributions; our multi-turn pressure prompts
   are off-distribution relative to pretraining text. May need to mix pressure
   dialogues into the averaging corpus.
4. Small n per cell (112–177 questions) — inherited from the paper; the same
   bootstrap discipline applies.

## 6. Framing for fellowship applications

One sentence: *"I found a weak, nonlinear, pre-behavioral signal of sycophantic
capitulation in open-weight LLMs; I now want to test whether it lives inside or
outside the model's verbalizable global workspace — either answer tells us
whether conversational alignment failures begin as deliberate or pre-verbal
processes, and whether they can be audited at the token level."*

This is directly aligned with interpretability-for-safety agendas (workspace
auditing, deception detection, monitoring before output).
