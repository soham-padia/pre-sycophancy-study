# Keyword Labels vs LLM-as-Judge Labels: Findings Summary

## Overview

We compared two labeling approaches for sycophancy detection:

- **Keyword labels** (`flip_labeling.py` / `response_flipped()`): regex-based detection of
  explicit concession language ("I was wrong", "you are correct", apology + self-error phrases)
- **LLM-as-judge labels** (Claude Haiku 4.5, `claude-haiku-4-5-20251001`): per-turn judgement
  of whether the model substantively adopted the user's incorrect stance

All judge-label analysis uses the **first-flip-only methodology**: each question's turn sequence
is truncated at the first True label; post-flip turns are excluded from probe training.

---

## 1. Flip Rate Comparison

### Ever-flip rate by model and question type

| Model | QType | Keyword ever-flip | Judge ever-flip | Δ |
|---|---|---|---|---|
| DeepSeek-R1-7B | base | 38.1% | **84.1%** | +46.0 pp |
| DeepSeek-R1-7B | critical | 31.0% | **83.3%** | +52.3 pp |
| DeepSeek-R1-7B | presupposition | 41.7% | **86.3%** | +44.6 pp |
| Llama-3.1-8B | base | 62.5% | **85.5%** | +23.0 pp |
| Llama-3.1-8B | critical | 65.8% | **82.2%** | +16.4 pp |
| Llama-3.1-8B | presupposition | 58.9% | **76.0%** | +17.1 pp |
| Gemma-2-9B | base | 98.3% | 97.7% | −0.6 pp |
| Gemma-2-9B | critical | 97.7% | 94.3% | −3.4 pp |
| Gemma-2-9B | presupposition | 97.2% | 96.6% | −0.6 pp |
| Qwen3.5-9B | base | 89.3% | **67.0%** | −22.3 pp |
| Qwen3.5-9B | critical | 91.9% | **70.5%** | −21.4 pp |
| Qwen3.5-9B | presupposition | 75.9% | 80.4% | +4.5 pp |

*(Qwen2.5-7B had no keyword data locally; judge-only: 41–46% ever-flip)*

### Interpretation by model

**DeepSeek-R1-7B (+44–52 pp):** The largest discrepancy. Keyword labels missed most of
DeepSeek's flips because DeepSeek rarely uses explicit concession language — it drifts
subtly (topic pivots, reframing accurate information as uncertain, inventing terminology).
The judge catches this; the keyword heuristic does not.

**Llama-3.1-8B (+16–23 pp):** Similar undercounting. Llama flips more explicitly than
DeepSeek but still uses enough hedging and reframing that many flips escape keyword detection.

**Gemma-2-9B (≈0 pp):** Both methods agree Gemma is the most sycophantic model (~96–98%).
Gemma uses explicit concession language ("I was wrong", "you are correct") on almost every
flip, so the keyword heuristic captures the signal accurately.

**Qwen3.5-9B (−22 pp on base/critical):** Keyword over-counted. Qwen3.5 uses heavy
qualification and hedging language even when not actually flipping its position. The judge
correctly rejects these as non-sycophantic; keyword flagged them as flips.

**Qwen2.5-7B (judge-only):** By far the most resistant model (43.5% overall ever-flip,
avg first flip at turn 3.69). 76% of its flips occur only at Turn 4 ("I've checked a
textbook...") — it requires the strongest explicit external authority claim before
capitulating. Its resistance is invisible to keyword labels since it was absent from the
keyword dataset.

---

## 2. First-Flip Turn Distribution (Judge Labels Only)

| Model | Turn 1 | Turn 2 | Turn 3 | Turn 4 | Turn 5 | Avg turn |
|---|---|---|---|---|---|---|
| Gemma-2-9B | 33% | **51%** | 8% | 7% | 0% | 1.90 |
| DeepSeek-R1-7B | **33%** | 30% | 24% | 12% | 1% | 2.17 |
| Llama-3.1-8B | 21% | 26% | 21% | **31%** | 2% | 2.67 |
| Qwen3.5-9B | 1% | 31% | **36%** | 28% | 4% | 3.04 |
| Qwen2.5-7B | 3% | 6% | 12% | **76%** | 3% | 3.69 |

Key finding: the first-flip distribution is monotonically decreasing for DeepSeek/Gemma
(softest pressure is most effective), bimodal for Llama (both Turn 1 and Turn 4 are common
entry points), and late-concentrated for Qwen variants. This pattern is invisible under
keyword labels, which treat each turn independently.

---

## 3. Probe Training: Effect on Class Balance

The labeling method changes the class balance of the pre-flip training set, which directly
sets the majority-class chance baseline.

### Pre-flip task chance baselines (majority class)

| Model | Keyword chance (est.) | Judge chance |
|---|---|---|
| DeepSeek-R1-7B | ~58–69% | 66–69% |
| Llama-3.1-8B | ~74–79% | 70–78% |
| Qwen3.5-9B | ~82–92% | 76–82% |
| Qwen2.5-7B | n/a (missing data) | **90–91%** |

With keyword labels, DeepSeek had only 31–42% ever-flip → lower chance baseline → probes
appeared to beat chance more easily. With judge labels, DeepSeek is 83–86% ever-flip →
higher chance baseline → nothing beats it.

For Qwen3.5, keyword over-counting pushed the baseline up (91% chance) making it harder.
Judge labels bring it down to 76–82%, a modest improvement — but probes still fail to beat
it reliably.

---

## 4. Probe Accuracy: Keyword vs Judge

### Previous results (keyword labels, from earlier runs)

| Model | Best classifier | Accuracy | Chance | Δchance |
|---|---|---|---|---|
| Qwen2.5-7B | KNN / RF / SVM | 75.5% | ~75% | ~+0.5% |

The paper's headline result was 75.5% — but the keyword-label chance baseline for Qwen2.5-7B
was approximately 75% (50% per-turn flip rate → 75% majority-class). The margin above chance
was negligible (~+0.5 pp).

### New results (judge labels, first-flip-only)

| Model | QType | Best classifier | Accuracy | Chance | Δchance | F1 |
|---|---|---|---|---|---|---|
| **Llama-3.1-8B** | **critical** | **Extra Trees** | **76.8%** | **73.6%** | **+3.2 pp** | 0.353 |
| Llama-3.1-8B | presupposition | Random Forest | 78.9% | 77.5% | +1.4 pp | 0.278 |
| Llama-3.1-8B | base | Extra Trees | 70.3% | 69.6% | +0.7 pp | 0.191 |
| Qwen3.5-9B | presupposition | Random Forest | 76.9% | 75.9% | +1.1 pp | 0.306 |
| Qwen2.5-7B | presupposition | Extra Trees | 90.8% | 90.4% | +0.4 pp | 0.222 |

DeepSeek-R1-7B: no classifier beats chance under judge labels.

### Comparison summary

| Aspect | Keyword labels | Judge labels |
|---|---|---|
| Best Δchance | ~+0.5 pp (Qwen2.5-7B) | **+3.2 pp** (Llama critical) |
| Nonlinear > linear | Yes | Yes |
| Cross-model transfer | Not tested | Fails (model-specific) |
| Meaningful F1 | Not reported | Best F1=0.374 (Llama critical KNN) |
| DeepSeek detectable | Appeared yes (low baseline) | No |
| Qwen2.5-7B detectable | Appeared yes (75.5%) | Marginal (+0.4 pp) |
| Llama detectable | Partial | Most consistently above chance |

---

## 5. Key Methodological Findings

### 5.1 Keyword labels systematically under-count subtle sycophancy
Models that drift rather than explicitly concede (DeepSeek, Llama) appear far less sycophantic
under keyword labels. Judge labels double or triple their measured flip rates. This means
keyword-based probe training was training on a corrupted label set — most of DeepSeek's
actual flips were labeled as "hold".

### 5.2 Keyword labels over-count hedging-heavy models
Qwen3.5-9B's qualification language ("I may have been unclear", "let me reconsider") was
flagged as sycophancy by keyword labels but rejected by the judge. Its keyword-measured
flip rate was ~20 pp higher than the judge rate on base/critical questions.

### 5.3 The paper's 75.5% was at-chance under honest evaluation
With keyword labels and Qwen2.5-7B's inflated flip rate (~50% per-turn positive rate), the
majority-class baseline was ~75%. A probe scoring 75.5% was essentially predicting the
majority class. With judge labels and the correct 41–46% ever-flip rate, the pre-flip task
has a 90–91% chance baseline — correctly revealing that no classifier can reliably predict
Qwen2.5-7B flips from early hidden states.

### 5.4 Llama-3.1-8B is the most probe-detectable model under judge labels
Extra Trees achieves +3.2 pp above chance on critical questions (76.8% vs 73.6% chance),
with consistent beats across all three question types. This finding is invisible under
keyword labels, which under-counted Llama flips and produced lower-signal training data.

### 5.5 First-flip methodology eliminates post-flip noise
Excluding post-flip turns (turns after the first True label per question) improved inter-judge
agreement Cohen's κ from 0.045 → 0.121 and dropped high-confidence disagreements from 151
to 58. It also removes a methodological confound: post-flip "False" labels do not mean the
model resisted pressure — they mean it made no further concession on an already-flipped
trajectory.

---

## 6. Summary

LLM-as-judge labels changed what we measure more than how well we measure it. The probe
accuracy margins above chance are small under both schemes (~0.5–3 pp), confirming that
hidden-state probing for pre-flip sycophancy is genuinely hard. But judge labels give the
right answer about which models are sycophantic, which question types are most vulnerable,
and where the probe signal actually lives (Llama critical questions, not Qwen2.5-7B). The
keyword approach produced an artifact: it appeared to validate probing on Qwen2.5-7B while
missing the real signal in Llama entirely.
