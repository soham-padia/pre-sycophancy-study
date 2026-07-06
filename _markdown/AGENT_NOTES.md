# Agent Notes — append-only session log

Convention: append a dated entry at the END of this file after any session that
changed something. Never edit or delete previous entries. Read the whole log
before starting work.

Template:

```markdown
## YYYY-MM-DD — <name / agent> (branch: <branch>)
- What changed:
- Files touched:
- Blocked on / needs review:
- Handoff notes:
```

---

## 2026-06-30 — Soham + Claude (branch: main)
- What changed:
  - Full code audit of the analysis pipeline. Three findings mattered:
    (1) base/critical/presupposition are system-prompt conditions on the same
    ~177 questions, not question types — paper §3.1 rewritten accordingly;
    (2) off-by-one in `plot_preflip_pca.py` and `plot_layer_sweep.py` (paired
    state-at-t with label-at-t instead of t+1) — fixed, figures regenerated,
    probe-failure result got *cleaner* (max acc = majority baseline exactly,
    all models); (3) Gemma base hidden-state file corrupt, critical/presup
    intact and unanalyzed.
  - Robustness checks added (`analysis/cosine_disruption_checks.py`):
    T1-exclusion (AUC survives) + bootstrap CIs + T0 completeness (100%).
  - Multi-turn AUC 0.5-floor bug fixed (`multiturn_auc_robustness.py`).
  - Paper revision pass applied in Overleaf tex (see `PAPER_STATUS.md` for the
    full done/remaining list vs. the professor's letter).
  - `download_data.py` added (HF: `sohampadianeu/pre-sycophancy-study-data`).
  - This `_markdown/` hub created.
- Files touched: `analysis/{multiturn_auc_robustness,plot_layer_sweep,plot_preflip_pca}.py`,
  `analysis_claude/{hidden_state_disruption,layer_sweep,preflip_pca}.png`,
  `analysis_claude/layer_sweep_accuracy.csv`, `_markdown/*` (commits 35a2478…b40e3ff+)
- Blocked on / needs review: nothing blocked. Remaining paper items in
  `PAPER_STATUS.md` §Remaining (permutation test is the priority; figure
  redesigns are Track 3).
- Handoff notes:
  - Regenerated figures need re-upload to Overleaf `new_figures/`
    (`hidden_state_disruption.png`, `preflip_pca.png`).
  - Two bib entries pending in Overleaf (see `PAPER_STATUS.md`).
  - Anyone writing new probe analyses: use the pre-flip pairing from
    `train_probes_v2.py::build_preflip_dataset` — see ONBOARDING "off-by-one trap".
