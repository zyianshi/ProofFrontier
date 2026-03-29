from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taam.downstream.minif2f import load_benchmark_tasks, write_benchmark_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Lean4 benchmark manifest from JSON/JSONL data")
    parser.add_argument("--data-path", type=str, required=True, help="Path to GAR data directory or a specific JSON/JSONL file")
    parser.add_argument("--benchmark-name", type=str, default="miniF2F", help="miniF2F | ProofNet")
    parser.add_argument("--split", type=str, default="test", help="valid | test")
    parser.add_argument("--task-limit", type=int, default=0)
    parser.add_argument("--out-jsonl", type=str, required=True)
    args = parser.parse_args()

    tasks = load_benchmark_tasks(
        Path(args.data_path),
        dataset_name=args.benchmark_name,
        split=args.split,
        task_limit=args.task_limit,
    )
    out_path = Path(args.out_jsonl)
    write_benchmark_manifest(tasks, out_path)
    summary = {
        "benchmark_name": args.benchmark_name,
        "data_path": args.data_path,
        "split": args.split,
        "task_limit": args.task_limit,
        "tasks": len(tasks),
        "out_jsonl": str(out_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
