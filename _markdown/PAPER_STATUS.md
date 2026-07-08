# Paper Status — EACL revision (deadline: August 2026)

Tracks the professor's revision letter (2026-05-25) item by item.
Last updated: 2026-07-08. Paper lives in Overleaf; latest local tex work was
synced from `_local/main.tex` (gitignored).

## Done ✅

| Item | Where |
|---|---|
| Figure 1 moved to page 2 (`figure*`), claim softened | teaser figure |
| T1-flip exclusion analysis (AUC survives: 0.630→0.646 Qwen2.5, 0.636→0.642 Llama) | §5.2 + `analysis/cosine_disruption_checks.py` |
| "Before any behavioral change" language sweep | throughout |
| Nonlinear contradiction fixed; "Why the keyword result is not the main result" para | abstract + §6.3 |
| Dataset accounting fixed — real design is ~177 questions × 3 system-prompt conditions (NOT 474 questions, NOT question types) | §3.1 rewritten; intro ~2,300 conversations; appendix 11,450 judge calls |
| Baseline correctness paragraph (+ judge-prompt mitigation: corrections not counted as flips) | §4 Baseline Position Assumption |
| Abstract / intro opening / 4 contribution bullets / conclusion (her versions) | throughout |
| Language sweep (confirms→supports, internal decision→representational state, etc.) | throughout |
| PCA caption "illustrating" not "confirming" + underlying off-by-one bug fixed, figure regenerated | §6.2 + `plot_preflip_pca.py` |
| Panel B condition mismatch fixed (now same conditions as Panel A), regenerated | `plot_layer_sweep.py`, commit b40e3ff |
| Multi-turn 0.5-floor bug fixed | `multiturn_auc_robustness.py` |
| Bootstrap 95% CIs reported; Limitations "no CIs" paragraph replaced | §5.2 + Limitations |
| Pressure schedule wording + intensity/act-type confound limitation + Qwen2.5 authority insight | §3.2, Limitations, Discussion |
| "Sycophancy is a trajectory" framing + 3-claim discussion | Discussion |
| Multiple comparisons + paired-conditions statement | Limitations |
| Global-workspace follow-up paragraph (J-space) | Discussion (Implications) |

## Remaining ❌

1. **Permutation test with layer-sweep correction** — shuffle labels, repeat the
   full layer sweep, compare observed peak to shuffled-peak distribution.
   Pure analysis of existing data (~30 min CPU). Professor pressed hardest here.
2. ~~Binomial CIs for flip rates in Table 2~~ — DONE 2026-07-08
   (`analysis/flip_rate_cis.py`; Table 2 in Overleaf updated with ±95% CI).
3. ~~Figure 2 redesign~~ — Tomas's PR #1 on `paper-jul-6` (stacked bars,
   keyword comparison to appendix). Pending review fixes before merge.
4. ~~Figure 3 → 4-panel evidence figure~~ — Tomas's PR #1. Pending review fixes:
   Panel C caption contradicts the paper's own CI values; generating scripts not
   committed; old Panel B (probe-by-layer) evidence needs a home in appendix:linear.
5. **One intro sentence**: why multi-turn pressure differs from single-turn
   preference conflict (her "four questions" — three are answered).

## Bib entries to add in Overleaf `custom.bib`

```bibtex
@inproceedings{yu-etal-2022-crepe,
    title = "{CREPE}: Open-Domain Question Answering with False Presuppositions",
    author = "Yu, Xinyan Velocity and Min, Sewon and Zettlemoyer, Luke and Hajishirzi, Hannaneh",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    year = "2023", address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics", pages = "10457--10480"
}

@misc{anthropic2026workspace,
    title = "Verbalizable Representations Form a Global Workspace in Language Models",
    author = "{Anthropic Interpretability Team}",
    year = "2026", howpublished = "Transformer Circuits Thread",
    url = "https://transformer-circuits.pub/2026/workspace/index.html"
}
```
(Verify the workspace paper's real author list before submission.)

## Submission checklist (learned the hard way)

- [ ] Fresh single-commit snapshot for the anonymous link (old history contains names)
- [ ] `_markdown/` excluded from the snapshot
- [ ] grep the snapshot for author names, `sohampadia`, `northeastern`, local paths
- [ ] No identifying filenames in figures/plots
- [ ] Figures in Overleaf `new_figures/` match latest regenerated versions
