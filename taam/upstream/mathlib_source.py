from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


DECL_RE = re.compile(
    r"^\s*(?:@[A-Za-z0-9_.]+\s+|(?:private|protected|noncomputable|unsafe|partial|local)\s+)*"
    r"(theorem|lemma)\s+([A-Za-z0-9_'.]+)\b"
)
IMPORT_RE = re.compile(r"^\s*import\s+([A-Za-z0-9_.]+)\s*$")
NAMESPACE_RE = re.compile(r"^\s*namespace\s+([A-Za-z0-9_.']+)\s*$")
SECTION_RE = re.compile(r"^\s*section\b")
END_RE = re.compile(r"^\s*end\b")


@dataclass
class MathlibTheoremRecord:
    theorem_id: str
    theorem_name: str
    declaration_kind: str
    module: str
    file_path: str
    line_start: int
    line_end: int
    imports: List[str] = field(default_factory=list)
    namespace_stack: List[str] = field(default_factory=list)
    declaration_header: str = ""
    declaration_text: str = ""
    source: str = "mathlib4_source_scan"
    metadata: Dict[str, str] = field(default_factory=dict)


def _module_name_from_path(path: Path, src_root: Path) -> str:
    rel = path.relative_to(src_root).with_suffix("")
    return ".".join(rel.parts)


def _collect_imports(lines: Sequence[str]) -> List[str]:
    imports: List[str] = []
    for line in lines:
        match = IMPORT_RE.match(line)
        if match:
            imports.append(match.group(1))
    return imports


def _find_declaration_ranges(lines: Sequence[str]) -> List[tuple[int, int, str, str, List[str]]]:
    ranges: List[tuple[int, int, str, str, List[str]]] = []
    namespace_stack: List[str] = []
    decl_indices: List[tuple[int, str, str, List[str]]] = []

    for idx, line in enumerate(lines):
        ns_match = NAMESPACE_RE.match(line)
        if ns_match:
            namespace_stack.append(ns_match.group(1))
            continue
        if SECTION_RE.match(line):
            namespace_stack.append("__section__")
            continue
        if END_RE.match(line):
            if namespace_stack:
                namespace_stack.pop()
            continue

        decl_match = DECL_RE.match(line)
        if decl_match:
            decl_indices.append(
                (idx, decl_match.group(1), decl_match.group(2), [ns for ns in namespace_stack if ns != "__section__"])
            )

    for pos, (start, kind, name, ns_stack) in enumerate(decl_indices):
        end = decl_indices[pos + 1][0] if pos + 1 < len(decl_indices) else len(lines)
        ranges.append((start, end, kind, name, ns_stack))
    return ranges


def _trim_trailing_block_lines(block_lines: Sequence[str]) -> List[str]:
    trimmed = list(block_lines)
    while trimmed and not trimmed[-1].strip():
        trimmed.pop()
    while trimmed and END_RE.match(trimmed[-1]):
        trimmed.pop()
        while trimmed and not trimmed[-1].strip():
            trimmed.pop()
    return trimmed


def scan_mathlib_file(path: Path, src_root: Path) -> List[MathlibTheoremRecord]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    imports = _collect_imports(lines)
    module_name = _module_name_from_path(path, src_root)
    records: List[MathlibTheoremRecord] = []

    for start, end, kind, name, namespace_stack in _find_declaration_ranges(lines):
        block_lines = _trim_trailing_block_lines(lines[start:end])
        header = block_lines[0].strip() if block_lines else ""
        record = MathlibTheoremRecord(
            theorem_id=f"{module_name}::{name}",
            theorem_name=name,
            declaration_kind=kind,
            module=module_name,
            file_path=str(path),
            line_start=start + 1,
            line_end=end,
            imports=list(imports),
            namespace_stack=list(namespace_stack),
            declaration_header=header,
            declaration_text="\n".join(block_lines).strip(),
        )
        records.append(record)
    return records


def scan_mathlib_source_tree(
    mathlib_root: Path,
    include_patterns: Optional[Sequence[str]] = None,
    exclude_patterns: Optional[Sequence[str]] = None,
    limit: int = 0,
) -> List[MathlibTheoremRecord]:
    src_candidates = [mathlib_root / "Mathlib", mathlib_root / "Mathlib4" / "Mathlib"]
    src_root = next((path for path in src_candidates if path.exists()), None)
    if src_root is None:
        raise FileNotFoundError(
            f"Could not find Mathlib source root under {mathlib_root}. Expected {mathlib_root / 'Mathlib'}."
        )

    include_patterns = list(include_patterns or ["**/*.lean"])
    exclude_patterns = list(exclude_patterns or [])
    files: List[Path] = []
    seen = set()
    for pattern in include_patterns:
        for path in sorted(src_root.glob(pattern)):
            if path.is_file() and path.suffix == ".lean" and path not in seen:
                seen.add(path)
                files.append(path)

    if exclude_patterns:
        excluded: set[Path] = set()
        for pattern in exclude_patterns:
            excluded.update(path for path in src_root.glob(pattern) if path.is_file())
        files = [path for path in files if path not in excluded]

    records: List[MathlibTheoremRecord] = []
    for path in files:
        records.extend(scan_mathlib_file(path, src_root=src_root))
        if limit > 0 and len(records) >= limit:
            return records[:limit]
    return records


def write_theorem_records_jsonl(records: Iterable[MathlibTheoremRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
