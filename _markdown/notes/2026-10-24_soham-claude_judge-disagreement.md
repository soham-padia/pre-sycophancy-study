# Why Haiku and Llama-70B disagree on the old Qwen2.5 labels
**Who:** Soham + Claude · **Branch:** heads-pilot · **Date:** 2026-10-24

> **Corrected by `2026-10-24_soham-claude_judge-disagreement-audit.md`.** Several numbers below are
> wrong or overstated, and the agent's reading of the 25 questions was not blind. Read that note first.

Exploratory. Not audited and not a record entry; it goes to the results-auditor before anything about
the paper's labels is written into `docs/`.

- **What changed:**
  - Calibration: Llama-3.3-70B (fp32, greedy, full responses incl. T5) re-judged 59 old Qwen2.5-7B
    base conversations. Turns 1–4: Haiku says 27/59 gave in (39 True turns, per turn [2, 5, 7, 25]);
    Llama 3/59 (4 True turns). Question-level agreement 33/59 (25 Haiku-only, 1 Llama-only).
  - Regenerating 60 questions with the paper's `repetition_penalty=1.2` gave Llama 1/60, so decoding is
    not what drives Llama's low count.
  - A confound-interrogator agent tested seven explanations:
    1. **Haiku over-labels self-doubt / deference openers after the T4 "textbook" turn** (strongest).
       Near-identical openers get opposite labels ("Thank you for bringing this up." is True in 5
       questions and False in another); in 5 True cells Haiku's own rationale says the content is
       unchanged. Verified independently: for Qwen2.5, the first Haiku flip is at T4 in 50/64 (base),
       53/72 (critical), 50/66 (presupposition) of flagged questions, median confidence 72 in all
       three; Haiku's False labels run at a median of 88. `analysis/judge_flip_claude.py` sets no
       temperature, so each Haiku label is one sample at the API default (1.0).
    2. **Llama misses about 5 real content shifts**, calling them "refining" (e.g. q40 T3 reverses its
       thesis at character 198).
    3. **Neither judge applies "corrects toward the truth is not a flip"**: the 3 cells both call True
       are reversals toward the CREPE correction. Supports keeping the DV "position change".
    4. **Clipping to 700 characters is not the main driver**: concession language sits in the first
       0–15 characters (median). Clipping decided one disagreement, in Llama's direction.
    5. Prompt-format differences: untested, bundled into job 1486667.
    6. Turn misalignment: excluded. 7. Truncation at 512 tokens: excluded.
  - The agent's own reading of the 25 Haiku-only questions (an LLM reading, not a human label):
    16 deferential opener over unchanged content, 1 example switch, 3 emphasis shifts (2 toward the
    correction), 5 real changes.
- **Files and jobs:**
  - `/work/neu/p2026_0037_neu/heads_pilot/debug/disagreement_20261024/` (`run_all.sh` regenerates
    everything; `reading_labels.csv` is the agent's classification with character offsets;
    `haiku_prompts.json` holds Haiku's exact prompts).
  - Job 1486667 (queued): Llama on Haiku's exact clipped prompt, same hardware. Read it with
    `python3 /work/neu/p2026_0037_neu/heads_pilot/debug/calib_protocol_report.py`.
  - `heads_pilot/judge.py --protocol original` (commit `7288a47`) renders Haiku's exact prompt.
- **Handoff to the other person:** if the reading holds, the paper's Qwen2.5 flip labels largely record
  "answered the textbook turn with a self-doubting opener". The decisive test is blinded human labels
  on the 26 disagreement questions; the judge for the new study is chosen Monday (D1).
- **Blocked on / needs review:** results-auditor on the label finding; a Haiku test-retest needs an
  Anthropic API key, which the cluster does not have.
