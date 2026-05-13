#!/usr/bin/env python3
"""Split an existing judge CSV into deterministic question shards.

This is intended for reusing already completed trajectory work when moving from
one serial output file to multiple shard-specific jobs.

Example:
    python analysis/split_judge_csv_shards.py \
      --input analysis/judge_outputs/ndif_judgements_trajectory.csv \
      --num-shards 3 \
      --models DeepSeek-R1-Distill-Qwen-7B \
      --output-prefix analysis/judge_outputs/deepseek_trajectory_shard
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path


FIELDNAMES = ["model", "question_type", "question", "turn", "judgement", "confidence", "rationale"]


def shard_for_question(model: str, question_type: str, question: str, num_shards: int) -> int:
    if num_shards <= 1:
        return 0
    key = f"{model}\n{question_type}\n{question}".encode("utf-8")
    digest = hashlib.sha1(key).hexdigest()
    return int(digest, 16) % num_shards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a judge CSV into deterministic question shards.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True, help="Writes <prefix><i>.csv for each shard")
    parser.add_argument("--models", nargs="*", default=None, help="Optional model filter")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1.")
    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    selected_models = set(args.models or [])
    grouped_rows = {i: [] for i in range(args.num_shards)}
    row_counts = Counter()
    question_counts = Counter()
    seen_questions = {i: set() for i in range(args.num_shards)}

    with args.input.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if selected_models and row["model"] not in selected_models:
                continue
            shard = shard_for_question(row["model"], row["question_type"], row["question"], args.num_shards)
            grouped_rows[shard].append({k: row.get(k, "") for k in FIELDNAMES})
            row_counts[shard] += 1
            qkey = (row["model"], row["question_type"], row["question"])
            if qkey not in seen_questions[shard]:
                seen_questions[shard].add(qkey)
                question_counts[shard] += 1

    for shard in range(args.num_shards):
        output_path = Path(f"{args.output_prefix}{shard}.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(grouped_rows[shard])
        print(
            f"shard={shard} rows={row_counts[shard]} questions={question_counts[shard]} "
            f"output={output_path}"
        )


if __name__ == "__main__":
    main()
