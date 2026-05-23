# Before the Model Caves: Detecting Pre-Capitulation States in Multi-Turn Sycophancy

> Submitted to EMNLP 2026 — Track: Interpretability and Analysis of Models for NLP

---

Large language models abandon correct answers under social pressure, not because new evidence arrives, but because users persist. This work shows what happens *inside the model before it does*. We construct a benchmark spanning five open-weight LLMs with monotonically escalating social pressure, extract layer-wise hidden states at every turn, and train probes to detect capitulation before it appears in outputs.

## Key Findings

1. **Early cosine disruption predicts eventual flips** — cosine similarity between Turn 0 and Turn 1 hidden states predicts whether a model will eventually capitulate, across all four probed models, without any classifier training (peak AUC 0.596–0.636).

2. **The signal is model-specific** — predictive layers differ substantially across architectures; cross-model probe transfer fails for all within-family pairs.

3. **Pre-flip geometry is nonlinear** — linear probes fail at every layer while nonlinear classifiers outperform them by 6.6 pp on average, with Fisher ratios near zero confirming the signal is not hyperplanar.

4. **L2 norm rises monotonically** — activation magnitude builds over 3–4 pressure turns before capitulation in three of four models.

### Hidden-State Trajectories (3D PCA)

Directional drift away from the T0 cluster is visible before any behavioral change occurs:

![Hidden-State Trajectories](analysis_claude/pca_3d.png)

### Zero-Shot Cosine Disruption Signal

AUC by layer for the best question type per model (A), and linear probe accuracy vs. majority-class chance (B):

![Cosine Disruption and Linear Probe Failure](analysis_claude/hidden_state_disruption.png)

### Pre-Flip Geometry (PCA Projection)

Pre-flip (red) and hold (blue) states intermix throughout the two leading principal components, confirming the absence of linear separability:

![Pre-Flip PCA](analysis_claude/preflip_pca.png)

### First-Flip Turn Distribution

Flip timing by model and question type under LLM-as-judge labels:

![Flip Turn Distribution](analysis_claude/flip_turn_distribution.png)

---

## Repository Structure

```
.
├── analysis/                  # Analysis and plotting scripts
│   ├── cosine_disruption.py   # Zero-shot cosine AUC sweep
│   ├── preflip_geometry.py    # Fisher ratio + LDA + classifier sweep
│   ├── multiturn_auc_robustness.py  # Signal persistence across turns (Appendix D)
│   ├── plot_hidden_state_disruption.py
│   ├── plot_preflip_pca.py
│   ├── plot_flip_turn_distribution.py
│   ├── plot_l2_norm_trend.py
│   └── ...
├── analysis_claude/           # Figures and CSVs from analysis runs
├── data/                      # Hidden states per model (gitignored — large)
│   ├── DeepSeek-R1-Distill-Qwen-7B/
│   ├── Llama-3.1-8B-Instruct/
│   ├── Qwen2.5-7B-Instruct/
│   └── Qwen3.5-9B/
├── debate_setting/            # Debate benchmark (original SYCON-Bench setting)
├── ethical-setting/           # Ethical benchmark setting
├── false-presuppositions-setting/  # False presuppositions benchmark setting
├── sbatch/                    # SLURM job scripts for HPC runs
├── flip_labeling.py           # Keyword-based flip detection
├── train_probes_v2.py         # Probe training (linear + nonlinear classifiers)
├── probe_model_comparison.py  # Cross-model probe evaluation
├── probe_judge_comparison.py  # Keyword vs. LLM-as-judge label comparison
└── requirements.txt
```

## Models Evaluated

| Model | Layers | Hidden Dim | Used For |
|-------|--------|------------|----------|
| DeepSeek-R1-7B-Distill | 29 | 3584 | Behavior + probing |
| Qwen2.5-7B-Instruct | 29 | 3584 | Behavior + probing |
| Llama-3.1-8B-Instruct | 33 | 4096 | Behavior + probing |
| Qwen3.5-9B | 33 | 4096 | Behavior + probing |
| Gemma-2-9B-it | 42 | 3584 | Behavioral analysis only |

## Setup

```bash
git clone <repo-url>
cd CS6120-Sycophancy-Detection
pip install -r requirements.txt
```

Requires access to NDIF for distributed hidden-state extraction via [NNsight](https://nnsight.net/).

## Running Experiments

### Generate multi-turn conversations

```bash
python analysis/generate_multiturn_dataset.py
```

### Run LLM-as-judge labeling

```bash
python analysis/judge_flip_claude.py
```

### Train probes

```bash
python train_probes_v2.py
```

### Reproduce analysis figures

```bash
python analysis/plot_hidden_state_disruption.py
python analysis/plot_preflip_pca.py
python analysis/plot_flip_turn_distribution.py
python analysis/plot_l2_norm_trend.py
python analysis/multiturn_auc_robustness.py
```

## Benchmark Settings

This repo extends [SYCON-Bench](https://github.com/JiseungHong/SYCON-Bench) with monotonically escalating pressure (each turn applies strictly stronger disagreement) and true multi-turn dialogue (full conversation history at every step). The three question categories — base, critical, and presupposition — test distinct resistance mechanisms.

## Citation

Citation will be added upon publication.
