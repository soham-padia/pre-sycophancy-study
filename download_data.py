#!/usr/bin/env python3
"""Download the pre-sycophancy-study data folder from HuggingFace Hub.

Usage:
    python download_data.py                  # full download (~22 GB incl. hidden states)
    python download_data.py --csv-only       # CSVs + JSONs only (~50 MB, no .pt files)
    python download_data.py --model Llama-3.1-8B-Instruct   # one model only

Set HF_REPO below to your HuggingFace dataset repo ID before sharing this script.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HF_REPO = "sohampadianeu/pre-sycophancy-study-data"   # ← set this before sharing

MODELS = [
    "DeepSeek-R1-Distill-Qwen-7B",
    "Gemma-2-9B",
    "Llama-3.1-8B-Instruct",
    "Qwen2.5-7B-Instruct",
    "Qwen3.5-9B",
]

QTYPES = ["base", "critical", "presupposition"]

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR  = REPO_ROOT / "data"


def build_file_list(models: list[str], csv_only: bool) -> list[str]:
    files = []
    for model in models:
        for qt in QTYPES:
            files.append(f"{model}/{qt}_multiturn.csv")
            files.append(f"{model}/{qt}_multiturn_metadata.json")
            files.append(f"{model}/{qt}_multiturn_hidden_states_summary.json")
            if not csv_only:
                files.append(f"{model}/{qt}_multiturn_hidden_states.pt")
        for i in range(5):
            files.append(f"{model}/prompt{i}.csv")
    return files


def download(args):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub not installed. Run: pip install huggingface-hub")
        sys.exit(1)

    if HF_REPO.startswith("TODO"):
        print("ERROR: Set HF_REPO in download_data.py before running.")
        sys.exit(1)

    models = [args.model] if args.model else MODELS
    files  = build_file_list(models, args.csv_only)

    print(f"Downloading {len(files)} files from {HF_REPO}")
    print(f"Destination: {DATA_DIR}\n")

    ok = skipped = failed = 0
    for rel in files:
        dest = DATA_DIR / rel
        if dest.exists():
            skipped += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            path = hf_hub_download(
                repo_id=HF_REPO,
                filename=rel,
                repo_type="dataset",
                local_dir=str(DATA_DIR),
            )
            print(f"  ✓  {rel}")
            ok += 1
        except Exception as e:
            print(f"  ✗  {rel}  ({e})")
            failed += 1

    print(f"\nDone: {ok} downloaded, {skipped} already present, {failed} failed.")
    if failed:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv-only", action="store_true",
                        help="Skip .pt hidden-state files (~22 GB); download CSVs only")
    parser.add_argument("--model", metavar="NAME",
                        help="Download a single model folder (e.g. Llama-3.1-8B-Instruct)")
    args = parser.parse_args()

    if args.model and args.model not in MODELS:
        print(f"Unknown model '{args.model}'. Choose from:\n  " + "\n  ".join(MODELS))
        sys.exit(1)

    download(args)


if __name__ == "__main__":
    main()
