from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

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


def _load_sample_json(path_str: str) -> Optional[Dict]:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _count_items(items: List[str], limit: int = 20) -> List[Dict[str, object]]:
    counter = Counter(item for item in items if item)
    return [{"value": key, "count": value} for key, value in counter.most_common(limit)]


def _distribution(values: List[int]) -> List[Dict[str, int]]:
    counter = Counter(values)
    return [{"value": value, "count": counter[value]} for value in sorted(counter)]


def _get_hidden_lemma_count(row: Dict[str, str], sample: Optional[Dict]) -> int:
    if row.get("hidden_lemma_count", ""):
        return int(row.get("hidden_lemma_count", 0) or 0)
    if sample is not None:
        return int(sample.get("hidden_lemma_count", len(sample.get("hidden_nodes", []))) or 0)
    return int(row.get("hidden_count", 0) or 0)


def _get_theorem_domain(row: Dict[str, str], sample: Optional[Dict]) -> str:
    return str(
        row.get("theorem_domain", "")
        or (sample or {}).get("theorem_domain", "")
        or (sample or {}).get("source_file_path", "")
    ).strip()


def _get_proof_source(row: Dict[str, str], sample: Optional[Dict]) -> str:
    return str(row.get("proof_source", "") or (sample or {}).get("proof_source", "")).strip()


def _enriched_rows(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    enriched: List[Dict[str, object]] = []
    for row in rows:
        sample = _load_sample_json(row.get("sample_json", ""))
        enriched.append(
            {
                "row": row,
                "sample": sample,
                "found_hard_sample": _to_bool(row.get("found_hard_sample", "")),
                "well_posed": None if row.get("well_posed", "") == "" else _to_bool(row.get("well_posed", "")),
                "hidden_lemma_count": _get_hidden_lemma_count(row, sample),
                "theorem_domain": _get_theorem_domain(row, sample),
                "proof_source": _get_proof_source(row, sample),
            }
        )
    return enriched


def analyze(rows: List[Dict[str, str]], group_fields: List[str]) -> Dict:
    enriched = _enriched_rows(rows)
    total = len(enriched)
    hard_rows = [r for r in enriched if r["found_hard_sample"]]
    hard_count = len(hard_rows)
    hard_rate = (hard_count / total) if total else 0.0

    hidden_counts = [int(r["hidden_lemma_count"]) for r in hard_rows]
    mean_hidden = mean(hidden_counts) if hidden_counts else 0.0

    wp_rows = [r for r in hard_rows if r["well_posed"] is not None]
    wp_vals = [bool(r["well_posed"]) for r in wp_rows]
    well_posed_rate = (sum(wp_vals) / len(wp_vals)) if wp_vals else None

    failed_counter = Counter(
        r["row"].get("failed_on", "") for r in hard_rows if r["row"].get("failed_on", "")
    )
    theorem_domains = [str(r["theorem_domain"]) for r in hard_rows if str(r["theorem_domain"]).strip()]
    proof_sources = [str(r["proof_source"]) for r in hard_rows if str(r["proof_source"]).strip()]

    grouped = defaultdict(list)
    for r in enriched:
        grouped[_group_key(r["row"], group_fields)].append(r)

    group_stats = []
    for key, grp in grouped.items():
        g_total = len(grp)
        g_hard = [r for r in grp if r["found_hard_sample"]]
        g_hidden = [int(r["hidden_lemma_count"]) for r in g_hard]
        g_wp = [r for r in g_hard if r["well_posed"] is not None]
        g_wp_vals = [bool(r["well_posed"]) for r in g_wp]
        g_domains = sorted({str(r["theorem_domain"]) for r in g_hard if str(r["theorem_domain"]).strip()})
        g_proof_sources = sorted({str(r["proof_source"]) for r in g_hard if str(r["proof_source"]).strip()})
        group_stats.append(
            {
                "group": key,
                "runs": g_total,
                "hard_samples": len(g_hard),
                "hard_rate": (len(g_hard) / g_total) if g_total else 0.0,
                "mean_hidden_lemma_count": mean(g_hidden) if g_hidden else 0.0,
                "well_posedness_rate": (sum(g_wp_vals) / len(g_wp_vals)) if g_wp_vals else None,
                "distinct_theorem_domains": len(g_domains),
                "distinct_proof_sources": len(g_proof_sources),
            }
        )
    group_stats.sort(key=lambda x: x["group"])

    return {
        "overall": {
            "runs": total,
            "hard_samples": hard_count,
            "hard_rate": hard_rate,
            "mean_hidden_lemma_count": mean_hidden,
            "well_posedness_rate": well_posed_rate,
            "distinct_theorem_domains": len(set(theorem_domains)),
            "distinct_proof_sources": len(set(proof_sources)),
        },
        "hidden_lemma_count_distribution": _distribution(hidden_counts),
        "theorem_domain_coverage": {
            "unique_domains": len(set(theorem_domains)),
            "top_domains": _count_items(theorem_domains),
        },
        "proof_source_coverage": {
            "unique_sources": len(set(proof_sources)),
            "top_sources": _count_items(proof_sources),
        },
        "top_failed_on": [{"node": k, "count": v} for k, v in failed_counter.most_common(10)],
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
    print("\nHidden lemma count distribution:")
    print(json.dumps(result["hidden_lemma_count_distribution"], ensure_ascii=False, indent=2))
    print("\nTheorem domain coverage:")
    print(json.dumps(result["theorem_domain_coverage"], ensure_ascii=False, indent=2))
    print("\nProof source coverage:")
    print(json.dumps(result["proof_source_coverage"], ensure_ascii=False, indent=2))
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
