# _markdown/notes/ — session notes, one file per entry

Replaces appending to `../AGENT_NOTES.md` (kept as the archive up to 2026-10-24): two people appending
to the same file at once produced merge conflicts, and a new file never conflicts.

- **Name:** `YYYY-MM-DD_<who>_<topic>.md`, e.g. `2026-10-26_utkarsh-claude_prereg-draft.md`.
- **When:** at the end of any session that changed something, or handed something to the other person.
- **Never edit someone else's note.** To correct one, write a new note that says what it corrects.
- **At the start of a session:** `git pull`, then read every note newer than your own last one
  (`ls -t _markdown/notes | head`).
- Notes are team communication, not the record. Numbers and claims enter the record only through
  `docs/` after review (see `CLAUDE.md`, "Verdict discipline").

Template:

```markdown
# <topic>
**Who:** <name / name + Claude> · **Branch:** <branch> · **Date:** YYYY-MM-DD

- **What changed:**
- **Files and jobs:** (paths, commits, Slurm job ids)
- **Handoff to the other person:** (what they can now do, by when)
- **Blocked on / needs review:**
```
