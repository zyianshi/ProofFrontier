from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taam.upstream.mathlib_source import scan_mathlib_source_tree, write_theorem_records_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch index theorem/lemma declarations from a local Mathlib4 source tree")
    parser.add_argument("--mathlib-root", type=str, required=True, help="Local path to Mathlib4 repository root")
    parser.add_argument(
        "--include",
        type=str,
        action="append",
        default=[],
        help="Glob relative to Mathlib source root, e.g. Algebra/**/*.lean. Can be repeated.",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        action="append",
        default=[],
        help="Exclude glob relative to Mathlib source root. Can be repeated.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-jsonl", type=str, required=True)
    args = parser.parse_args()

    records = scan_mathlib_source_tree(
        Path(args.mathlib_root),
        include_patterns=args.include or ["**/*.lean"],
        exclude_patterns=args.exclude or [],
        limit=args.limit,
    )
    out_path = Path(args.out_jsonl)
    write_theorem_records_jsonl(records, out_path)
    summary = {
        "mathlib_root": args.mathlib_root,
        "theorems_indexed": len(records),
        "out_jsonl": str(out_path),
        "include": args.include or ["**/*.lean"],
        "exclude": args.exclude or [],
        "limit": args.limit,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
