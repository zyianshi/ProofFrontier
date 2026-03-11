from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _to_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def _load_rows(summary_files: List[Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for p in summary_files:
        with p.open("r", encoding="utf-8", newline="") as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


def _collect_summary_files(args: argparse.Namespace) -> List[Path]:
    files: List[Path] = []
    for p in args.summary_csv:
        files.append(Path(p))
    if args.sweeps_root:
        files.extend(sorted(Path(args.sweeps_root).glob("*_summary.csv")))
    return files


def _group_key(row: Dict[str, str], fields: List[str]) -> str:
    return " | ".join(f"{k}={row.get(k, '')}" for k in fields)


def analyze(rows: List[Dict[str, str]], group_fields: List[str]) -> Dict:
    total = len(rows)
    hard_rows = [r for r in rows if _to_bool(r.get("found_hard_sample", ""))]
    hard_count = len(hard_rows)
    hard_rate = (hard_count / total) if total else 0.0

    hidden_counts = [int(r.get("hidden_count", 0) or 0) for r in hard_rows]
    mean_hidden = mean(hidden_counts) if hidden_counts else 0.0

    wp_rows = [r for r in hard_rows if r.get("well_posed", "") != ""]
    wp_vals = [_to_bool(r.get("well_posed", "")) for r in wp_rows]
    well_posed_rate = (sum(wp_vals) / len(wp_vals)) if wp_vals else None

    failed_counter = Counter(r.get("failed_on", "") for r in hard_rows if r.get("failed_on", ""))
    top_failed = failed_counter.most_common(10)

    grouped = defaultdict(list)
    for r in rows:
        grouped[_group_key(r, group_fields)].append(r)

    group_stats = []
    for key, grp in grouped.items():
        g_total = len(grp)
        g_hard = [r for r in grp if _to_bool(r.get("found_hard_sample", ""))]
        g_hidden = [int(r.get("hidden_count", 0) or 0) for r in g_hard]
        g_wp = [r for r in g_hard if r.get("well_posed", "") != ""]
        g_wp_vals = [_to_bool(r.get("well_posed", "")) for r in g_wp]
        group_stats.append(
            {
                "group": key,
                "runs": g_total,
                "hard_samples": len(g_hard),
                "hard_rate": (len(g_hard) / g_total) if g_total else 0.0,
                "mean_hidden_count": mean(g_hidden) if g_hidden else 0.0,
                "well_posed_rate": (sum(g_wp_vals) / len(g_wp_vals)) if g_wp_vals else None,
            }
        )
    group_stats.sort(key=lambda x: x["group"])

    return {
        "overall": {
            "runs": total,
            "hard_samples": hard_count,
            "hard_rate": hard_rate,
            "mean_hidden_count": mean_hidden,
            "well_posed_rate": well_posed_rate,
        },
        "top_failed_on": [{"node": k, "count": v} for k, v in top_failed],
        "group_by": group_fields,
        "groups": group_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze TAAM sweep summary CSVs")
    parser.add_argument(
        "--summary-csv",
        type=str,
        action="append",
        default=[],
        help="Summary CSV path. Can be passed multiple times.",
    )
    parser.add_argument("--sweeps-root", type=str, default="", help="Folder containing *_summary.csv files")
    parser.add_argument(
        "--group-by",
        type=str,
        default="masking_strategy,solver_type",
        help="Comma-separated fields for grouped stats",
    )
    parser.add_argument("--out-json", type=str, default="", help="Optional output JSON path")
    args = parser.parse_args()

    files = _collect_summary_files(args)
    if not files:
        raise SystemExit("No summary CSV provided. Use --summary-csv or --sweeps-root.")
    rows = _load_rows(files)
    if not rows:
        raise SystemExit("No rows loaded from summary CSVs.")

    group_fields = [x.strip() for x in args.group_by.split(",") if x.strip()]
    result = analyze(rows, group_fields=group_fields)

    print("Overall:")
    print(json.dumps(result["overall"], ensure_ascii=False, indent=2))
    print("\nTop failed_on nodes:")
    print(json.dumps(result["top_failed_on"], ensure_ascii=False, indent=2))
    print("\nGrouped:")
    print(json.dumps(result["groups"], ensure_ascii=False, indent=2))

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved analysis: {out_path.resolve()}")


if __name__ == "__main__":
    main()
