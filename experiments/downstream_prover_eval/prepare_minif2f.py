from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taam.downstream.minif2f import load_miniF2F_tasks, write_miniF2F_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Lean4 miniF2F manifest from GAR-style JSON/JSONL data")
    parser.add_argument("--data-path", type=str, required=True, help="Path to GAR data directory or a specific JSON/JSONL file")
    parser.add_argument("--split", type=str, default="test", help="valid | test")
    parser.add_argument("--task-limit", type=int, default=0)
    parser.add_argument("--out-jsonl", type=str, required=True)
    args = parser.parse_args()

    tasks = load_miniF2F_tasks(Path(args.data_path), split=args.split, task_limit=args.task_limit)
    out_path = Path(args.out_jsonl)
    write_miniF2F_manifest(tasks, out_path)
    summary = {
        "data_path": args.data_path,
        "split": args.split,
        "task_limit": args.task_limit,
        "tasks": len(tasks),
        "out_jsonl": str(out_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
