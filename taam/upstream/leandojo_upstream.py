from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


_TRACED_THEOREM_INDEX: Dict[int, Dict[str, Any]] = {}


DECL_HEAD_RE = re.compile(
    r"^\s*(?:@[A-Za-z0-9_.]+\s+|(?:private|protected|noncomputable|unsafe|partial|local)\s+)*"
    r"(theorem|lemma)\s+([A-Za-z0-9_.']+)"
)


def _require_leandojo() -> Tuple[Any, Any, Any]:
    try:
        from lean_dojo import LeanGitRepo, trace
        from lean_dojo.data_extraction.traced_data import TracedRepo
    except ImportError as exc:
        raise RuntimeError(
            "LeanDojo is required for real upstream extraction. Install `lean-dojo` and its Lean dependencies first."
        ) from exc
    return LeanGitRepo, trace, TracedRepo


def _strip_decl_suffix(text: str) -> str:
    cleaned = text.strip()
    if cleaned.endswith(":="):
        cleaned = cleaned[:-2].rstrip()
    if cleaned.endswith("where"):
        cleaned = cleaned[: -len("where")].rstrip()
    return cleaned


def _find_top_level_colon(text: str) -> int:
    depth_round = 0
    depth_square = 0
    depth_curly = 0
    for idx, ch in enumerate(text):
        if ch == "(":
            depth_round += 1
        elif ch == ")":
            depth_round = max(0, depth_round - 1)
        elif ch == "[":
            depth_square += 1
        elif ch == "]":
            depth_square = max(0, depth_square - 1)
        elif ch == "{":
            depth_curly += 1
        elif ch == "}":
            depth_curly = max(0, depth_curly - 1)
        elif ch == ":" and depth_round == 0 and depth_square == 0 and depth_curly == 0:
            return idx
    return -1


def parse_declaration_header(header: str) -> Dict[str, Any]:
    """
    Parse Lean theorem/lemma declaration headers from LeanDojo's `get_theorem_statement()`.

    Example input:
      theorem foo (a b : Nat) : a + b = b + a :=
    """
    text = _strip_decl_suffix(header)
    match = DECL_HEAD_RE.match(text)
    if not match:
        return {
            "declaration_kind": "",
            "declaration_name": "",
            "context_chunks": [],
            "target_statement": text,
            "premise_type": text,
            "raw_header": header,
        }

    kind, decl_name = match.group(1), match.group(2)
    rest = text[match.end() :].strip()
    colon_idx = _find_top_level_colon(rest)
    if colon_idx < 0:
        context_prefix = rest.strip()
        target_statement = rest.strip()
    else:
        context_prefix = rest[:colon_idx].strip()
        target_statement = rest[colon_idx + 1 :].strip()

    context_chunks = [context_prefix] if context_prefix else []
    premise_type = (
        target_statement if not context_prefix else f"∀ {context_prefix}, {target_statement}"
    )
    return {
        "declaration_kind": kind,
        "declaration_name": decl_name,
        "context_chunks": context_chunks,
        "target_statement": target_statement,
        "premise_type": premise_type,
        "raw_header": header,
    }


def _theorem_identity(traced_theorem: Any) -> Dict[str, str]:
    theorem = traced_theorem.theorem
    return {
        "full_name": str(theorem.full_name),
        "file_path": str(theorem.file_path),
        "uid": str(getattr(theorem, "uid", "")),
        "uhash": str(getattr(theorem, "uhash", "")),
    }


def create_repo_spec(
    repo_url: str = "",
    commit: str = "",
    local_repo_path: str = "",
) -> Any:
    LeanGitRepo, _trace, _TracedRepo = _require_leandojo()
    if repo_url and commit:
        return LeanGitRepo(repo_url, commit)
    if local_repo_path:
        return LeanGitRepo.from_path(local_repo_path)
    raise ValueError("Provide repo_url + commit, or a working local_repo_path")


def trace_repo_to_disk(
    repo_url: str = "",
    commit: str = "",
    local_repo_path: str = "",
    dst_dir: str = "",
    build_deps: bool = True,
) -> Any:
    repo = create_repo_spec(repo_url=repo_url, commit=commit, local_repo_path=local_repo_path)
    _LeanGitRepo, trace, _TracedRepo = _require_leandojo()
    return trace(repo, dst_dir=dst_dir or None, build_deps=build_deps)


def load_traced_repo(root_dir: str, build_deps: bool = True) -> Any:
    _LeanGitRepo, _trace, TracedRepo = _require_leandojo()
    try:
        return TracedRepo.load_from_disk(root_dir, build_deps=build_deps)
    except Exception:
        return TracedRepo.from_traced_files(root_dir, build_deps=build_deps)


def traced_theorem_to_inventory_record(traced_theorem: Any) -> Dict[str, Any]:
    identity = _theorem_identity(traced_theorem)
    header = traced_theorem.get_theorem_statement()
    parsed = parse_declaration_header(header)
    tactic_proof = traced_theorem.get_tactic_proof() or ""
    premise_names = sorted(set(str(x) for x in traced_theorem.get_premise_full_names()))

    return {
        "theorem_id": identity["full_name"],
        "full_name": identity["full_name"],
        "file_path": identity["file_path"],
        "uid": identity["uid"],
        "uhash": identity["uhash"],
        "declaration_kind": parsed["declaration_kind"],
        "declaration_name": parsed["declaration_name"],
        "statement_header": header,
        "target_statement": parsed["target_statement"],
        "theorem_context": parsed["context_chunks"],
        "premise_type": parsed["premise_type"],
        "has_tactic_proof": bool(traced_theorem.has_tactic_proof()),
        "tactic_proof": tactic_proof,
        "num_tactics": int(traced_theorem.get_num_tactics()),
        "premise_full_names": premise_names,
        "source": "leandojo_traced_repo",
    }


def export_theorem_inventory(
    traced_repo: Any,
    out_jsonl: Path,
    require_tactic_proof: bool = True,
    min_tactics: int = 1,
) -> int:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_jsonl.open("w", encoding="utf-8") as f:
        for traced_theorem in traced_repo.get_traced_theorems():
            if require_tactic_proof and not traced_theorem.has_tactic_proof():
                continue
            if int(traced_theorem.get_num_tactics()) < min_tactics:
                continue
            record = traced_theorem_to_inventory_record(traced_theorem)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


@dataclass
class DependencyGraphBuildStats:
    target_theorem: str
    resolved_nodes: int
    edges: int
    skipped_non_theorem_premises: int
    truncated_by_depth: int
    truncated_by_node_limit: bool


def _resolve_traced_theorem(traced_repo: Any, full_name: str) -> Optional[Any]:
    try:
        return traced_repo.get_traced_theorem(full_name)
    except Exception:
        pass

    cache = _TRACED_THEOREM_INDEX.get(id(traced_repo))
    if cache is None:
        cache = {}
        try:
            traced_theorems = list(traced_repo.get_traced_theorems())
        except Exception:
            traced_theorems = []
        for traced_theorem in traced_theorems:
            theorem = traced_theorem.theorem
            theorem_full_name = str(getattr(theorem, "full_name", ""))
            theorem_uid = str(getattr(theorem, "uid", ""))
            if theorem_full_name:
                cache.setdefault(theorem_full_name, traced_theorem)
                cache.setdefault(theorem_full_name.split(".")[-1], traced_theorem)
            if theorem_uid:
                cache.setdefault(theorem_uid, traced_theorem)
                cache.setdefault(theorem_uid.split(":")[-1], traced_theorem)
        _TRACED_THEOREM_INDEX[id(traced_repo)] = cache

    return cache.get(full_name)


def build_taam_graph_from_traced_theorem(
    traced_repo: Any,
    theorem_full_name: str,
    max_depth: int = 2,
    max_nodes: int = 128,
) -> Tuple[Dict[str, Any], DependencyGraphBuildStats]:
    target_theorem = _resolve_traced_theorem(traced_repo, theorem_full_name)
    if target_theorem is None:
        raise ValueError(f"Cannot resolve theorem `{theorem_full_name}` in traced repo")

    nodes: Dict[str, Dict[str, Any]] = {}
    edges: set[Tuple[str, str]] = set()
    skipped_non_theorem_premises = 0
    truncated_by_depth = 0
    truncated_by_node_limit = False
    deps_cache: Dict[str, List[str]] = {}

    def visit(full_name: str, depth: int) -> None:
        nonlocal skipped_non_theorem_premises, truncated_by_depth, truncated_by_node_limit
        if full_name in nodes:
            return
        if len(nodes) >= max_nodes:
            truncated_by_node_limit = True
            return

        traced_theorem = _resolve_traced_theorem(traced_repo, full_name)
        if traced_theorem is None:
            skipped_non_theorem_premises += 1
            return

        header = traced_theorem.get_theorem_statement()
        parsed = parse_declaration_header(header)
        identity = _theorem_identity(traced_theorem)
        tactic_proof = traced_theorem.get_tactic_proof() or ""
        premise_names = sorted(set(str(x) for x in traced_theorem.get_premise_full_names()) - {full_name})
        deps_cache[full_name] = premise_names

        nodes[full_name] = {
            "id": full_name,
            "kind": "lemma",
            "statement": parsed["target_statement"],
            "lean_statement": parsed["premise_type"],
            "difficulty": 0.5,
            "metadata": {
                "file_path": identity["file_path"],
                "uid": identity["uid"],
                "uhash": identity["uhash"],
                "declaration_header": header,
                "proof_completion": tactic_proof,
                "proof_source": "leandojo_tactic_proof",
                "num_tactics": str(int(traced_theorem.get_num_tactics())),
            },
        }

        if depth >= max_depth:
            truncated_by_depth += len(premise_names)
            return

        for dep_name in premise_names:
            dep_theorem = _resolve_traced_theorem(traced_repo, dep_name)
            if dep_theorem is None:
                skipped_non_theorem_premises += 1
                continue
            edges.add((dep_name, full_name))
            visit(dep_name, depth + 1)

    visit(theorem_full_name, depth=0)

    # Classify target + leaf premises.
    target_header = target_theorem.get_theorem_statement()
    target_parsed = parse_declaration_header(target_header)
    if theorem_full_name not in nodes:
        raise RuntimeError(f"Target theorem `{theorem_full_name}` was not added to the graph")
    nodes[theorem_full_name]["kind"] = "target"
    nodes[theorem_full_name]["statement"] = target_parsed["target_statement"]
    nodes[theorem_full_name]["lean_statement"] = target_parsed["target_statement"]

    has_incoming: Dict[str, bool] = {node_id: False for node_id in nodes}
    for src, dst in edges:
        if dst in has_incoming:
            has_incoming[dst] = True
        if src not in nodes:
            continue
    for node_id, node in nodes.items():
        if node["kind"] == "target":
            continue
        if not has_incoming.get(node_id, False):
            node["kind"] = "premise"

    graph = {
        "theorem_id": theorem_full_name,
        "target_id": theorem_full_name,
        "nodes": [nodes[node_id] for node_id in sorted(nodes)],
        "edges": [[src, dst] for src, dst in sorted(edges)],
        "imports": ["Mathlib"],
        "theorem_context": target_parsed["context_chunks"],
    }
    stats = DependencyGraphBuildStats(
        target_theorem=theorem_full_name,
        resolved_nodes=len(nodes),
        edges=len(edges),
        skipped_non_theorem_premises=skipped_non_theorem_premises,
        truncated_by_depth=truncated_by_depth,
        truncated_by_node_limit=truncated_by_node_limit,
    )
    return graph, stats


def write_jsonl(records: Iterable[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(obj: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")





