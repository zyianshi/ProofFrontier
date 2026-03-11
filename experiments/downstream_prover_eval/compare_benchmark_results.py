from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taam.downstream.downstream import compare_benchmark_runs, load_benchmark_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare before/after downstream benchmark results")
    parser.add_argument("--before", type=str, action="append", required=True, help="Before benchmark result file")
    parser.add_argument("--after", type=str, action="append", required=True, help="After benchmark result file")
    parser.add_argument("--out-json", type=str, default="", help="Optional output path")
    args = parser.parse_args()

    before_rows = load_benchmark_rows([Path(p) for p in args.before])
    after_rows = load_benchmark_rows([Path(p) for p in args.after])
    result = compare_benchmark_runs(before_rows, after_rows)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
