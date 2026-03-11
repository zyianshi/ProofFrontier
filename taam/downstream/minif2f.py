from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


THEOREM_NAME_KEYS = ("theorem_name", "name", "decl_name", "id", "task_id")
SPLIT_KEYS = ("split", "subset", "partition")
LEAN_CODE_KEYS = ("lean_code", "lean4_code", "formal_code", "code", "src")
TARGET_KEYS = ("formal_statement", "lean_statement", "statement", "goal", "target")
IMPORT_KEYS = ("imports", "header", "preamble")
CONTEXT_KEYS = ("theorem_context", "context", "variables")
PROOF_KEYS = ("proof", "proof_completion", "tactic")


@dataclass
class MiniF2FTask:
    task_id: str
    theorem_name: str
    split: str
    source_file: str
    lean_code: str
    language: str = "lean4"
    dataset: str = "miniF2F-lean4"
    metadata: Dict[str, Any] = field(default_factory=dict)


def _first_nonempty(record: Dict[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = record.get(key, "")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _normalize_split(split: str) -> str:
    s = split.lower().strip()
    if s in {"validation", "val"}:
        return "valid"
    return s or "test"


def _read_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "records", "examples", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Unsupported Lean4 miniF2F data format: {path}")


def _collect_candidate_files(data_path: Path, split: str) -> List[Path]:
    split_norm = _normalize_split(split)
    if data_path.is_file():
        return [data_path]

    search_roots = [data_path]
    if (data_path / "data").exists():
        search_roots.insert(0, data_path / "data")

    candidates: List[Path] = []
    for root in search_roots:
        for pattern in ("*.json", "*.jsonl"):
            candidates.extend(sorted(root.glob(pattern)))
    if not candidates:
        raise FileNotFoundError(f"No JSON/JSONL files found under {data_path}")

    preferred = [
        path
        for path in candidates
        if "minif2f" in path.name.lower() and split_norm in path.name.lower()
    ]
    if preferred:
        return preferred

    split_filtered = [path for path in candidates if split_norm in path.name.lower()]
    if split_filtered:
        return split_filtered

    minif2f_only = [path for path in candidates if "minif2f" in path.name.lower()]
    if minif2f_only:
        return minif2f_only
    return candidates


def _stringify_imports(imports_value: Any) -> List[str]:
    if isinstance(imports_value, list):
        return [str(item).strip() for item in imports_value if str(item).strip()]
    if isinstance(imports_value, str) and imports_value.strip():
        text = imports_value.strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if all(line.startswith("import ") for line in lines):
            return [line.removeprefix("import ").strip() for line in lines]
        return [text]
    return []


def _stringify_context(context_value: Any) -> List[str]:
    if isinstance(context_value, list):
        return [str(item).strip() for item in context_value if str(item).strip()]
    if isinstance(context_value, str) and context_value.strip():
        return [context_value.strip()]
    return []


def _build_lean_code(record: Dict[str, Any], theorem_name: str) -> str:
    explicit_code = _first_nonempty(record, LEAN_CODE_KEYS)
    if explicit_code:
        return explicit_code if explicit_code.endswith("\n") else explicit_code + "\n"

    target = _first_nonempty(record, TARGET_KEYS)
    if not target:
        raise ValueError(f"Cannot derive Lean4 miniF2F task without Lean code or target statement: {record}")

    imports = _stringify_imports(record.get("imports", record.get("header", "")))
    context = _stringify_context(record.get("theorem_context", record.get("context", "")))
    proof = _first_nonempty(record, PROOF_KEYS)

    lines: List[str] = []
    for imp in imports:
        if imp.startswith("import "):
            lines.append(imp)
        else:
            lines.append(f"import {imp}")
    if lines:
        lines.append("")

    signature = " ".join(context).strip()
    theorem_decl = f"theorem {theorem_name}"
    if signature:
        theorem_decl += f" {signature}"
    theorem_decl += f" : {target} := by"
    lines.append(theorem_decl)
    lines.append(f"  {proof}" if proof else "  sorry")
    return "\n".join(lines) + "\n"


def _record_matches_split(record: Dict[str, Any], split: str) -> bool:
    split_norm = _normalize_split(split)
    record_split = _normalize_split(_first_nonempty(record, SPLIT_KEYS))
    if not record_split:
        return True
    return record_split == split_norm


def _record_to_task(record: Dict[str, Any], source_file: Path, split: str, index: int) -> MiniF2FTask:
    theorem_name = _first_nonempty(record, THEOREM_NAME_KEYS) or f"{_normalize_split(split)}_{index}"
    task_id = str(record.get("task_id", f"miniF2F-lean4::{_normalize_split(split)}::{theorem_name}"))
    lean_code = _build_lean_code(record, theorem_name)
    metadata = dict(record)
    return MiniF2FTask(
        task_id=task_id,
        theorem_name=theorem_name,
        split=_normalize_split(_first_nonempty(record, SPLIT_KEYS) or split),
        source_file=str(source_file),
        lean_code=lean_code,
        metadata=metadata,
    )


def load_miniF2F_tasks(data_path: Path, split: str = "test", task_limit: int = 0) -> List[MiniF2FTask]:
    tasks: List[MiniF2FTask] = []
    files = _collect_candidate_files(data_path, split)
    for file_path in files:
        records = _read_records(file_path)
        for idx, record in enumerate(records, start=1):
            if not isinstance(record, dict):
                continue
            if not _record_matches_split(record, split):
                continue
            try:
                tasks.append(_record_to_task(record, file_path, split=split, index=idx))
            except ValueError:
                continue
    if task_limit > 0:
        tasks = tasks[:task_limit]
    return tasks


def write_miniF2F_manifest(tasks: Iterable[MiniF2FTask], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(asdict(task), ensure_ascii=False) + "\n")


def load_manifest(path: Path) -> List[MiniF2FTask]:
    tasks: List[MiniF2FTask] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            tasks.append(MiniF2FTask(**data))
    return tasks
