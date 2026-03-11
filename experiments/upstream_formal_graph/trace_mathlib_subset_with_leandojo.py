from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taam.upstream.leandojo_upstream import export_theorem_inventory
from taam.upstream.mathlib_source import MathlibTheoremRecord, scan_mathlib_source_tree


def _run(cmd: Sequence[str], cwd: Path) -> None:
    subprocess.run(list(cmd), cwd=str(cwd), check=True, text=True)


def _resolve_extractor_path() -> Path:
    from lean_dojo.data_extraction.trace import LEAN4_DATA_EXTRACTOR_PATH

    return Path(LEAN4_DATA_EXTRACTOR_PATH)


def _select_files(
    records: Sequence[MathlibTheoremRecord], theorem_limit: int, max_files: int
) -> tuple[List[Dict[str, object]], int]:
    theorem_count_by_file: "OrderedDict[str, int]" = OrderedDict()
    for record in records:
        theorem_count_by_file.setdefault(record.file_path, 0)
        theorem_count_by_file[record.file_path] += 1

    selected: List[Dict[str, object]] = []
    total_theorems = 0
    for file_path, count in theorem_count_by_file.items():
        selected.append({"file_path": file_path, "theorem_count": count})
        total_theorems += count
        if theorem_limit > 0 and total_theorems >= theorem_limit:
            break
        if max_files > 0 and len(selected) >= max_files:
            break
    return selected, total_theorems


def _patch_extractor(extractor_dst: Path) -> None:
    text = extractor_dst.read_text(encoding="utf-8")
    text = re.sub(r"(?m)^\s*assert! .*path\.pathExists\s*$\n?", "", text, count=1)
    extractor_dst.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace a small Mathlib4 subset with LeanDojo ExtractData and export theorem inventory"
    )
    parser.add_argument("--mathlib-root", type=str, required=True)
    parser.add_argument("--include", type=str, action="append", default=[])
    parser.add_argument("--exclude", type=str, action="append", default=[])
    parser.add_argument("--theorem-limit", type=int, default=100)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--inventory-jsonl", type=str, required=True)
    parser.add_argument("--selected-files-json", type=str, default="")
    parser.add_argument("--allow-non-tactic", action="store_true", default=False)
    parser.add_argument("--min-tactics", type=int, default=1)
    parser.add_argument("--skip-cache-get", action="store_true", default=False)
    parser.add_argument("--keep-extractor", action="store_true", default=False)
    args = parser.parse_args()

    mathlib_root = Path(args.mathlib_root).resolve()
    include_patterns = args.include or ["Algebra/**/*.lean"]
    records = scan_mathlib_source_tree(
        mathlib_root,
        include_patterns=include_patterns,
        exclude_patterns=args.exclude or [],
        limit=0,
    )
    selected_files, covered_theorems = _select_files(
        records, args.theorem_limit, args.max_files
    )
    if not selected_files:
        raise SystemExit("No source files matched the selection criteria")

    extractor_src = _resolve_extractor_path()
    extractor_dst = mathlib_root / "ExtractData.lean"
    shutil.copyfile(extractor_src, extractor_dst)
    _patch_extractor(extractor_dst)

    if not args.skip_cache_get:
        _run(["lake", "exe", "cache", "get"], cwd=mathlib_root)

    for row in selected_files:
        file_path = Path(str(row["file_path"])).resolve()
        rel_path = file_path.relative_to(mathlib_root)
        _run(
            ["lake", "env", "lean", "--run", "ExtractData.lean", str(rel_path)],
            cwd=mathlib_root,
        )

    from lean_dojo.data_extraction.traced_data import TracedRepo

    traced_repo = TracedRepo.from_traced_files(mathlib_root, build_deps=False)
    inventory_count = export_theorem_inventory(
        traced_repo,
        Path(args.inventory_jsonl),
        require_tactic_proof=not args.allow_non_tactic,
        min_tactics=args.min_tactics,
    )

    if args.selected_files_json:
        selected_out = Path(args.selected_files_json)
        selected_out.parent.mkdir(parents=True, exist_ok=True)
        selected_out.write_text(
            json.dumps(
                {
                    "mathlib_root": str(mathlib_root),
                    "include": include_patterns,
                    "exclude": args.exclude or [],
                    "theorem_limit": args.theorem_limit,
                    "covered_theorems": covered_theorems,
                    "selected_files": selected_files,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    if not args.keep_extractor and extractor_dst.exists():
        extractor_dst.unlink()

    summary = {
        "mathlib_root": str(mathlib_root),
        "inventory_jsonl": args.inventory_jsonl,
        "inventory_count": inventory_count,
        "selected_file_count": len(selected_files),
        "covered_theorems": covered_theorems,
        "include": include_patterns,
        "exclude": args.exclude or [],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

