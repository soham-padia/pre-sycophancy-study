# EMNLP Revision Task Split

Four tracks. Each person owns their track end to end — writing, code, and figure edits where applicable. Target: full draft ready for Prof. Malihe review by end of next week.

---

## Track 1 — Writing: Abstract, Introduction, Conclusion (Soham)

**Goal:** Make the paper's framing honest and crisp. The paper's best version is "sycophancy has a representational history, but it is weak, model-specific, and not linearly structured."

**Tasks:**

1. **Rewrite the abstract** using Prof. Malihe's suggested version as a base. Key constraints:
   - AUC range stays 0.596–0.636 (do not round up)
   - Remove "before any behavioral change" unless Track 2 analysis supports it
   - Add: "nonlinear classifiers improve over linear probes, but absolute predictive value remains weak under semantic judge labels"

2. **Rewrite the introduction** around the four questions:
   - Why output metrics are too late
   - Why multi-turn pressure is different from single-turn preference conflict
   - Why hidden states are the right place to look
   - Why the result is surprising: signal exists but is weak, nonlinear, model-specific
   - Opening line: "The moment a model gives in is not necessarily the moment the failure begins..."

3. **Cut contributions to 4 bullets** (see professor's list — do not claim reliable detection)

4. **Rewrite the conclusion** using Prof. Malihe's suggested version. Land on three claims:
   - Sycophancy has temporal structure
   - Internal monitoring is possible but fragile
   - The geometry is not universal

5. **Language sweep across full paper** — find and replace:
   - "the internal decision to capitulate" → "a representational state associated with later capitulation"
   - "detects pre-capitulation states" → "provides an early weak signal of capitulation risk"
   - "confirms" → "supports" or "is consistent with"
   - "linear probes fail, proving nonlinear geometry" → "linear probes fail under our setup; nonlinear classifiers recover somewhat stronger relative structure"
   - Every instance of "before any output change" or "before any behavioral change" — flag for Track 2 to confirm or soften

---

## Track 2 — Analysis: Statistics and the T1-Flip Fix (Soham / whoever ran the probes)

**Goal:** Fix the central vulnerability (T1-flip confound) and add statistical rigor. No new model inference needed — all existing hidden states and labels.

**Tasks:**

1. **T1-flip exclusion AUC** *(already done — `analysis/cosine_disruption_checks.py`)*
   - Results: AUC holds or improves after exclusion for Qwen2.5 and Llama
   - Write up the result as a paragraph in §5.2: "Among cases that do not behaviorally flip at T1, the first pressure turn still predicts later capitulation..."
   - If AUC drops for a model, soften the claim for that model specifically

2. **Bootstrap 95% CIs** *(already done — same script)*
   - Add CI values to every reported AUC in the paper (Table or inline)
   - Note which cases are strictly above chance vs. marginal

3. **Permutation test for peak AUC** *(new — 1 hour of work)*
   - For each (model, qtype), shuffle labels 1000 times, rerun layer sweep, record peak AUC from shuffled labels
   - Report: "observed peak AUC = X; permutation p = Y"
   - This defends against the "you searched over layers and reported the max" concern

4. **Fix nonlinear classifier contradiction**
   - Abstract says +6.6pp. §6.3 says no classifier exceeds majority baseline by >2.5pp under judge labels
   - Resolution: the +6.6pp is from keyword labels; the 2.5pp cap is from judge labels
   - In abstract, write: "Nonlinear classifiers consistently outperform linear probes, but absolute predictive performance remains weak under semantic judge labels"
   - Move keyword label classifier result to appendix with a clear label
   - Add a short paragraph: "Why the keyword result is not the main result" — keyword labels capture explicit concession language; judge labels capture semantic position shifts; the latter is the real alignment problem

5. **Fix dataset accounting**
   - Paper says 474 questions total but compute appendix implies 474 per category
   - Verify the actual number: total questions vs. per-category, and fix every instance in the paper

---

## Track 3 — Figures: Redesign Figure 3 and Clean Up Others (Tomas / Vedant)

**Goal:** Reviewers should understand the story in 10 seconds from the figures alone.

**Tasks:**

1. **Figure 3 — redesign as 4-panel central evidence figure:**
   - Panel A: T0→T1 cosine AUC by layer, one line per model (current Panel A)
   - Panel B: same after removing T1-first-flip cases (new — use Track 2 results)
   - Panel C: best layer per model, shown as a bar chart with CI bars
   - Panel D: drift magnitude vs. predictive AUC scatter — showing the dissociation (a model can have high drift and low AUC)
   - Add bootstrap CI bands or error bars to Panels A and B

2. **Figure 1 — schematic:**
   - Move to page 2 (professor's explicit request)
   - Replace "before any output change" with "early pressure state predicts later flip above chance" (conditional on Track 2 result)

3. **Figure 2 — flip turn distribution:**
   - Use judge labels only in the main figure
   - Move keyword vs. judge comparison (current `flip_turn_comparison.png`) to appendix
   - Consider replacing bubble chart with stacked bar chart — simpler and easier to parse for reviewers

4. **Figure 4 — PCA:**
   - Change caption from "visually confirming" lack of linear separability to "illustrating"
   - The actual evidence is Fisher ratios and probe performance — caption must say this
   - Move 3D PCA trajectories and L2 norm plots to appendix

---

## Track 4 — Sections: Dataset, Related Work, Limitations (Utkarsh / Vedant)

**Goal:** Fix the structural gaps that a careful reviewer will catch.

**Tasks:**

1. **Baseline correctness section** (new paragraph in §3 or §4, titled "Baseline Correctness Filtering"):
   - If T0 filtering WAS done: describe the prompt and how it was verified
   - If T0 filtering was NOT done (current situation): add the following and remove all "abandons a correct answer" language:
     > "Our labels measure shifts away from the model's initial factual position, not always shifts away from a verified correct answer. We therefore interpret the results as pressure-induced position change rather than correctness loss in all cases."
   - Remove every instance of "abandons a correct answer" as a universal claim

2. **Related work — reframe as a debate:**
   - Prior sycophancy work: measures output behavior only
   - Prior probing work: assumes linear recoverability
   - Prior steering work: suggests sycophancy may have linear directions
   - This paper: early pressure signal is different — weak, nonlinear, model-specific
   - Do not just list papers. Frame each paragraph around where this paper agrees or disagrees.

3. **Pressure schedule clarification:**
   - Replace "strictly stronger disagreement" with "a fixed escalation schedule from mild doubt to authority-backed contradiction"
   - In limitations: note that the schedule confounds pressure intensity with social act type (doubt vs. external authority vs. dominance)
   - In discussion: note that Qwen2.5's late flips (at the textbook turn) suggest authority cues may matter more than raw contradiction for some models

4. **Statistical uncertainty additions:**
   - Add confidence intervals to flip rates in Table 2 (binomial CIs are fine: `p ± 1.96 * sqrt(p(1-p)/n)`)
   - Add a statement about multiple comparisons in §5 or limitations
   - Coordinate with Track 2 for AUC CIs

---

## Git Setup and Data

### Clone and install
```bash
git clone https://github.com/soham-padia/pre-sycophancy-study.git
cd pre-sycophancy-study
pip install -r requirements.txt
```

### What IS in the repo (clone gives you this)
- All analysis scripts (`analysis/`, `analysis_claude/`)
- Benchmark generation code (`false-presuppositions-setting/`, `debate_setting/`)
- Judge outputs (`analysis_claude/*_judgements*.csv`)
- Figures (`analysis_claude/*.png`)
- Flip labeling (`flip_labeling.py`)
- Probe training (`train_probes_v2.py`)

### What is NOT in the repo (large files, get from Soham's Drive/cluster)
- `data/` — multiturn CSV files (~50MB per model, 5 models)
- `data/*/` — hidden state `.pt` files (~1–4GB per model per question type)

**Ask Soham for the Google Drive link** for `data/`. Download and place it at repo root so paths resolve correctly.

### Do NOT download / run locally
- Model weights (7–9B params each, need 16GB+ VRAM) — all extraction already done
- SLURM sbatch scripts (`sbatch/`) — cluster-only, not needed for analysis

### Key scripts for revision work
| Task | Script |
|---|---|
| T1-exclusion + bootstrap CIs | `analysis/cosine_disruption_checks.py` |
| Cosine AUC by layer (raw data) | generates `analysis_claude/cosine_disruption.csv` |
| Figure 3 (Panel A+B currently) | `analysis/plot_hidden_state_disruption.py` |
| Figure 2 (flip turn distribution) | `analysis/plot_flip_turn_distribution.py` |
| Figure 4 (PCA) | `analysis/plot_preflip_pca.py` |
| Probe training sweep | `train_probes_v2.py` |

### Running the analysis
```bash
# Robustness checks (T1-exclusion + bootstrap CIs)
python analysis/cosine_disruption_checks.py

# Regenerate figures
python analysis/plot_hidden_state_disruption.py
python analysis/plot_flip_turn_distribution.py
python analysis/plot_preflip_pca.py
```
All outputs go to `analysis_claude/`.
