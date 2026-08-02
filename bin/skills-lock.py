#!/usr/bin/env python3
"""skills-lock.py — pin and verify the PROCESS LAYER (agents, skills, standing law).

The constitution polices the scientific record; this polices the constitution. A lockfile of
content hashes makes process drift detectable: a silently weakened falsifier contract, a
checklist item that vanished, a hook/linter edit nobody reviewed. Git tracks changes; the lock
distinguishes "deliberate, reviewed process change" (regenerate + commit, dated) from "drift
since the last review" (verify fails).

Usage:
  python bin/skills-lock.py generate     # (re)write skills-lock.json — a DELIBERATE act:
                                         #   do it only when a process change has been reviewed
  python bin/skills-lock.py verify       # exit 0 if process layer matches the lock; else exit 1
                                         #   and print per-file drift (added / removed / modified)

Locked paths (relative to the lockfile's directory): .claude/agents/**, .claude/skills/**,
CLAUDE.md (or CLAUDE.md.template). Living docs (taxonomy, errata, citable) are deliberately NOT
locked — they are append-with-date by design; locking them would punish the loop working.
Stdlib only. The lock records tool versions of nothing and hashes of everything.
"""
import hashlib, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "skills-lock.json"
GLOBS = [".claude/agents/*.md", ".claude/skills/*/SKILL.md", "CLAUDE.md", "CLAUDE.md.template"]


def current_hashes():
    out = {}
    for g in GLOBS:
        for p in sorted(ROOT.glob(g)):
            out[str(p.relative_to(ROOT))] = "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def generate():
    lock = {"_comment": "Process-layer lock. Regenerating this file is a DELIBERATE, reviewed act "
                        "— commit it with a message saying WHAT process change it blesses.",
            "generated": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "files": current_hashes()}
    LOCK.write_text(json.dumps(lock, indent=2) + "\n")
    print(f"locked {len(lock['files'])} process files -> {LOCK.name}")


def verify():
    if not LOCK.exists():
        print("NO LOCK: skills-lock.json missing — run `generate` after reviewing the process layer")
        return 1
    locked = json.loads(LOCK.read_text())["files"]
    now = current_hashes()
    added = sorted(set(now) - set(locked))
    removed = sorted(set(locked) - set(now))
    modified = sorted(k for k in set(now) & set(locked) if now[k] != locked[k])
    if not (added or removed or modified):
        print(f"PROCESS LAYER INTACT: {len(now)} files match the lock "
              f"(generated {json.loads(LOCK.read_text())['generated']})")
        return 0
    print("PROCESS DRIFT DETECTED — either revert, or review + regenerate the lock deliberately:")
    for k in modified: print(f"  MODIFIED  {k}")
    for k in added:    print(f"  ADDED     {k} (unlocked process file)")
    for k in removed:  print(f"  REMOVED   {k}")
    return 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if cmd == "generate":
        generate()
    elif cmd == "verify":
        sys.exit(verify())
    else:
        sys.exit(f"unknown command {cmd!r} (use: generate | verify)")
