from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taam.upstream.leandojo_upstream import (  # noqa: E402
    DependencyGraphBuildStats,
    build_taam_graph_from_traced_theorem,
    load_traced_repo,
    write_json,
)


def _load_inventory(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _select_inventory_rows(
    inventory_rows: List[Dict],
    module_prefixes: List[str],
    require_tactic_proof: bool,
    min_tactics: int,
    limit: int,
) -> List[Dict]:
    selected: List[Dict] = []
    for row in inventory_rows:
        if require_tactic_proof and not bool(row.get("has_tactic_proof", False)):
            continue
        if int(row.get("num_tactics", 0)) < min_tactics:
            continue
        full_name = str(row.get("full_name", row.get("theorem_id", "")))
        file_path = str(row.get("file_path", ""))
        if module_prefixes and not any(
            full_name.startswith(prefix)
            or file_path.replace("\\", "/").find(prefix.replace(".", "/")) >= 0
            for prefix in module_prefixes
        ):
            continue
        selected.append(row)
        if limit > 0 and len(selected) >= limit:
            break
    return selected


def _row_key_candidates(row: Dict) -> List[str]:
    keys: List[str] = []
    for raw in (
        row.get("full_name", ""),
        row.get("theorem_id", ""),
        row.get("uid", ""),
        row.get("declaration_name", ""),
    ):
        text = str(raw or "").strip()
        if not text:
            continue
        keys.append(text)
        if ":" in text:
            keys.append(text.split(":")[-1])
        if "." in text:
            keys.append(text.split(".")[-1])
    deduped: List[str] = []
    seen = set()
    for key in keys:
        if key and key not in seen:
            seen.add(key)
            deduped.append(key)
    return deduped


def _canonical_name(row: Dict) -> str:
    for key in (
        str(row.get("full_name", "")).strip(),
        str(row.get("theorem_id", "")).strip(),
        str(row.get("declaration_name", "")).strip(),
    ):
        if key:
            return key
    raise ValueError("inventory row is missing theorem identifiers")


def _index_inventory_rows(inventory_rows: List[Dict]) -> Dict[str, Dict]:
    index: Dict[str, Dict] = {}
    for row in inventory_rows:
        row["_canonical_name"] = _canonical_name(row)
        for key in _row_key_candidates(row):
            index.setdefault(key, row)
    return index


def _build_taam_graph_from_inventory_row(
    inventory_index: Dict[str, Dict],
    target_row: Dict,
    max_depth: int,
    max_nodes: int,
) -> Tuple[Dict, DependencyGraphBuildStats]:
    theorem_full_name = str(target_row["_canonical_name"])
    nodes: Dict[str, Dict] = {}
    edges: set[Tuple[str, str]] = set()
    skipped_non_theorem_premises = 0
    truncated_by_depth = 0
    truncated_by_node_limit = False

    def visit(row: Dict, depth: int) -> None:
        nonlocal skipped_non_theorem_premises, truncated_by_depth, truncated_by_node_limit
        node_id = str(row["_canonical_name"])
        if node_id in nodes:
            return
        if len(nodes) >= max_nodes:
            truncated_by_node_limit = True
            return

        premise_names = sorted(
            set(str(x) for x in row.get("premise_full_names", []))
            - {
                node_id,
                str(row.get("full_name", "")),
                str(row.get("theorem_id", "")),
                str(row.get("declaration_name", "")),
            }
        )

        nodes[node_id] = {
            "id": node_id,
            "kind": "lemma",
            "statement": str(row.get("target_statement", "")),
            "lean_statement": str(row.get("premise_type", row.get("target_statement", ""))),
            "difficulty": 0.5,
            "metadata": {
                "file_path": str(row.get("file_path", "")),
                "uid": str(row.get("uid", "")),
                "uhash": str(row.get("uhash", "")),
                "declaration_header": str(row.get("statement_header", "")),
                "proof_completion": str(row.get("tactic_proof", "")),
                "proof_source": "leandojo_tactic_proof",
                "num_tactics": str(int(row.get("num_tactics", 0))),
            },
        }

        if depth >= max_depth:
            truncated_by_depth += len(premise_names)
            return

        for dep_name in premise_names:
            dep_row = inventory_index.get(dep_name)
            if dep_row is None:
                skipped_non_theorem_premises += 1
                continue
            dep_id = str(dep_row["_canonical_name"])
            edges.add((dep_id, node_id))
            visit(dep_row, depth + 1)

    visit(target_row, depth=0)

    if theorem_full_name not in nodes:
        raise RuntimeError(f"Target theorem `{theorem_full_name}` was not added to the graph")

    target_statement = str(target_row.get("target_statement", ""))
    nodes[theorem_full_name]["kind"] = "target"
    nodes[theorem_full_name]["statement"] = target_statement
    nodes[theorem_full_name]["lean_statement"] = target_statement

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
        "theorem_context": list(target_row.get("theorem_context", [])),
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build TAAM graph corpus from a LeanDojo traced Mathlib4 repo"
    )
    parser.add_argument("--traced-repo-root", type=str, required=True)
    parser.add_argument("--inventory-jsonl", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument(
        "--module-prefix",
        type=str,
        action="append",
        default=[],
        help="Restrict exported graphs to theorem names or file paths under these prefixes",
    )
    parser.add_argument("--allow-non-tactic", action="store_true", default=False)
    parser.add_argument("--min-tactics", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-nodes", type=int, default=128)
    parser.add_argument("--no-build-deps", action="store_true", default=False)
    args = parser.parse_args()

    inventory_rows = _load_inventory(Path(args.inventory_jsonl))
    selected_rows = _select_inventory_rows(
        inventory_rows,
        module_prefixes=args.module_prefix,
        require_tactic_proof=not args.allow_non_tactic,
        min_tactics=args.min_tactics,
        limit=args.limit,
    )

    traced_repo = None
    inventory_index: Dict[str, Dict] = {}
    if args.no_build_deps:
        inventory_index = _index_inventory_rows(inventory_rows)
    else:
        traced_repo = load_traced_repo(args.traced_repo_root, build_deps=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: List[Dict] = []
    for row in selected_rows:
        theorem_name = str(row.get("full_name", row.get("theorem_id", "")))
        try:
            if args.no_build_deps:
                graph, stats = _build_taam_graph_from_inventory_row(
                    inventory_index,
                    row,
                    max_depth=args.max_depth,
                    max_nodes=args.max_nodes,
                )
            else:
                graph, stats = build_taam_graph_from_traced_theorem(
                    traced_repo,
                    theorem_full_name=theorem_name,
                    max_depth=args.max_depth,
                    max_nodes=args.max_nodes,
                )
        except Exception as exc:
            manifest_rows.append(
                {
                    "theorem_id": theorem_name,
                    "status": "error",
                    "error": str(exc),
                }
            )
            continue

        safe_name = theorem_name.replace(".", "__")
        graph_path = out_dir / f"{safe_name}.graph.json"
        write_json(graph, graph_path)
        manifest_rows.append(
            {
                "theorem_id": theorem_name,
                "status": "ok",
                "graph_json": str(graph_path),
                "resolved_nodes": stats.resolved_nodes,
                "edges": stats.edges,
                "skipped_non_theorem_premises": stats.skipped_non_theorem_premises,
                "truncated_by_depth": stats.truncated_by_depth,
                "truncated_by_node_limit": stats.truncated_by_node_limit,
            }
        )

    manifest_path = out_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "traced_repo_root": args.traced_repo_root,
        "inventory_jsonl": args.inventory_jsonl,
        "exported": sum(1 for row in manifest_rows if row["status"] == "ok"),
        "failed": sum(1 for row in manifest_rows if row["status"] != "ok"),
        "out_dir": str(out_dir),
        "manifest_jsonl": str(manifest_path),
        "max_depth": args.max_depth,
        "max_nodes": args.max_nodes,
        "build_deps": not args.no_build_deps,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
