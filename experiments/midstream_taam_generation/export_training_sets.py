from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _to_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def _load_rows(summary_files: Iterable[Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in summary_files:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


def _load_sample(path_str: str) -> Dict | None:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_strategy(strategy: str) -> str:
    strategy = strategy.strip().lower()
    if strategy == "greedy_topology":
        return "topology"
    if strategy == "low_centrality_first":
        return "low_centrality"
    if strategy == "random_mask":
        return "random"
    return strategy or "unknown"


def _iter_records(rows: List[Dict[str, str]], only_well_posed: bool) -> Iterable[Dict]:
    for row in rows:
        if not _to_bool(row.get("found_hard_sample", "")):
            continue
        if only_well_posed and row.get("well_posed", "") and not _to_bool(row["well_posed"]):
            continue

        sample = _load_sample(row.get("sample_json", ""))
        if sample is None:
            continue

        yield {
            "theorem_id": sample.get("theorem_id", row.get("theorem_id", "")),
            "target_id": sample.get("target_id", row.get("target_id", "")),
            "masking_strategy": _normalize_strategy(row.get("masking_strategy", sample.get("masking_strategy", ""))),
            "failed_on": sample.get("failed_on", row.get("failed_on", "")),
            "hidden_nodes": sample.get("hidden_nodes", []),
            "visible_nodes": sample.get("visible_nodes", []),
            "hidden_lemma_count": sample.get("hidden_lemma_count", row.get("hidden_lemma_count", 0)),
            "visible_lemma_count": sample.get("visible_lemma_count", row.get("visible_lemma_count", 0)),
            "theorem_domain": sample.get("theorem_domain", row.get("theorem_domain", "")),
            "source_file_path": sample.get("source_file_path", row.get("source_file_path", "")),
            "source_graph_size": sample.get("source_graph_size", row.get("source_graph_size", 0)),
            "source_graph_edge_count": sample.get("source_graph_edge_count", row.get("source_graph_edge_count", 0)),
            "proof_source": sample.get("proof_source", row.get("proof_source", "")),
            "proof_completion": sample.get("proof_completion", ""),
            "lean_problem": sample.get("lean_problem", ""),
            "full_lean_problem": sample.get("full_lean_problem", ""),
            "well_posed": sample.get("well_posed", row.get("well_posed", "")),
            "sample_json": row.get("sample_json", ""),
            "out_dir": row.get("out_dir", ""),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export first-pass TAAM training sets by masking strategy")
    parser.add_argument(
        "--summary-csv",
        type=str,
        action="append",
        default=[],
        help="Summary CSV path. Can be passed multiple times.",
    )
    parser.add_argument("--sweeps-root", type=str, default="", help="Folder containing *_summary.csv files")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/midstream_taam_generation/training_sets",
        help="Output directory for per-strategy JSONL corpora",
    )
    parser.add_argument(
        "--include-non-well-posed",
        action="store_true",
        help="Keep hard samples even if well_posed=false.",
    )
    args = parser.parse_args()

    summary_files = [Path(p) for p in args.summary_csv]
    if args.sweeps_root:
        summary_files.extend(sorted(Path(args.sweeps_root).glob("*_summary.csv")))
    if not summary_files:
        raise SystemExit("No summary CSV provided. Use --summary-csv or --sweeps-root.")

    rows = _load_rows(summary_files)
    records = list(_iter_records(rows, only_well_posed=not args.include_non_well_posed))
    if not records:
        raise SystemExit("No exportable hard samples found.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped: Dict[str, List[Dict]] = {}
    for record in records:
        grouped.setdefault(record["masking_strategy"], []).append(record)

    manifest = {"total_records": len(records), "strategies": {}}
    for strategy, strategy_records in sorted(grouped.items()):
        out_path = out_dir / f"{strategy}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for record in strategy_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        manifest["strategies"][strategy] = {
            "records": len(strategy_records),
            "output_jsonl": str(out_path.resolve()),
            "theorem_domains": dict(Counter(r["theorem_domain"] for r in strategy_records if r["theorem_domain"])),
            "proof_sources": dict(Counter(r["proof_source"] for r in strategy_records if r["proof_source"])),
        }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Saved training sets: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
