from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_theorem_lookup(records: Iterable[Dict], key: str = "theorem_id") -> Dict[str, Dict]:
    lookup: Dict[str, Dict] = {}
    for record in records:
        lookup[str(record[key])] = record
    return lookup


def enrich_trace_with_inventory(trace_record: Dict, theorem_record: Dict) -> Dict:
    enriched = dict(trace_record)
    enriched.setdefault("theorem_id", theorem_record.get("theorem_id", ""))
    enriched.setdefault("imports", theorem_record.get("imports", ["Mathlib"]))
    enriched.setdefault("metadata", {})
    enriched["metadata"] = {
        **dict(theorem_record.get("metadata", {})),
        **dict(enriched.get("metadata", {})),
        "module": theorem_record.get("module", ""),
        "file_path": theorem_record.get("file_path", ""),
        "line_start": theorem_record.get("line_start", ""),
        "line_end": theorem_record.get("line_end", ""),
        "declaration_kind": theorem_record.get("declaration_kind", ""),
        "source_inventory": theorem_record.get("source", "mathlib4_source_scan"),
    }
    return enriched


def write_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
