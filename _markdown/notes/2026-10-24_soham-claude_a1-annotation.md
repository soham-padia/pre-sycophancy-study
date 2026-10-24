# A1: 34 conversations to label this weekend, both of us, blind
**Who:** Soham + Claude · **Branch:** heads-pilot · **Date:** 2026-10-24

- **What changed:**
  - The annotation page now holds sample A1: 34 old Qwen2.5 base conversations, shuffled. It mixes the
    26 questions where Haiku and Llama-70B disagree with 8 controls where they agree. Which is which is
    hidden until both of us finish.
  - The analysis plan is locked: `docs/prereg/A1_judge_adjudication.md` (commit `f06c894`). It fixes
    what gets computed and what each outcome would mean before any label exists.
- **Handoff to the other person (Utkarsh):**
  - Open the annotation page (link in `_markdown/UTKARSH_START_HERE.md`). If it says it can't save, ask
    Soham for Contributor access.
  - For each conversation: label turns 1–5 against what the model said just before, then reveal the
    reference answer and mark whether the first answer handled the false assumption, then submit.
    The codebook is on the page; "Moved to the truth" wins over "Gave in" when both fit. About 4
    minutes each, so roughly 2.5 hours in total.
  - Label before reading `2026-10-24_soham-claude_judge-disagreement.md`: it states the hypothesis under
    test, and Soham has already read it.
  - Until both of us have submitted all 34: don't discuss labels with each other, and don't open
    `/work/.../heads_pilot/calib_old_qwen25_base/`, `/work/.../heads_pilot/debug/`, or the agent's
    `reading_labels.csv`. They show which conversations are which.
  - When you're done, write a note here saying so. Claude then computes the plan's outcomes.
- **Files and jobs:**
  - Sample builder, items and manifest: `/work/neu/p2026_0037_neu/heads_pilot/annotation_a1/` (the key
    file is readable by its owner only until labelling ends).
  - Still running: job 1486667 (Llama on Haiku's exact clipped prompt) and job 1486853 (the same prompt
    with clipping off). Their comparison lands in
    `/work/neu/p2026_0037_neu/heads_pilot/debug/calib_protocol_report.txt`; don't read it before you finish labelling.
  - A results-auditor pass is recomputing the judge-disagreement numbers.
- **Blocked on / needs review:** nothing; this needs only our labels.
