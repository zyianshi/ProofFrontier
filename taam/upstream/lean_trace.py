from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from .extraction import FormalGraphExtractor
from ..types import GraphNode, NodeId, TAAMGraph, make_empty_adj


class LeanTraceExtractor:
    """
    Phase 1 adapter:
    Convert Lean/LeanDojo-style trace JSON into TAAMGraph.

    Supported input schemas:
    1) graph-like:
       {
         "theorem_id": "...",
         "target_id": "T",
         "nodes": [{"id","kind","statement","difficulty","metadata"}],
         "edges": [["A","B"], ...]
       }

    2) trace-like:
       {
         "theorem_id": "...",
         "target": {"id":"T","statement":"..."},
         "premises": [{"id":"P1","statement":"..."}],
         "lemmas": [{"id":"L1","statement":"...","depends_on":["P1"]}],
         "target_depends_on": ["L1","P1"]
       }
    """

    @staticmethod
    def from_json(path: Path) -> TAAMGraph:
        data = json.loads(path.read_text(encoding="utf-8"))
        if "nodes" in data and "edges" in data and "target_id" in data:
            # Already close to graph format.
            return FormalGraphExtractor.from_json(path)
        return LeanTraceExtractor._from_trace_schema(data)

    @staticmethod
    def _from_trace_schema(data: Dict) -> TAAMGraph:
        theorem_id = str(data.get("theorem_id", "unknown_theorem"))
        target = data.get("target", {})
        target_id = str(target.get("id", "T"))
        target_stmt = str(target.get("statement", "target theorem"))
        target_lean = str(target.get("lean_statement", target_stmt))

        nodes: Dict[NodeId, GraphNode] = {}
        nodes[target_id] = GraphNode(
            target_id,
            "target",
            target_stmt,
            lean_statement=target_lean,
            difficulty=0.9,
            metadata=dict(target.get("metadata", {})),
        )

        for p in data.get("premises", []):
            pid = str(p["id"])
            nodes[pid] = GraphNode(
                node_id=pid,
                kind="premise",
                statement=str(p.get("statement", "")),
                lean_statement=str(p.get("lean_statement", p.get("statement", ""))),
                difficulty=float(p.get("difficulty", 0.2)),
                metadata=dict(p.get("metadata", {})),
            )

        for l in data.get("lemmas", []):
            lid = str(l["id"])
            nodes[lid] = GraphNode(
                node_id=lid,
                kind="lemma",
                statement=str(l.get("statement", "")),
                lean_statement=str(l.get("lean_statement", l.get("statement", ""))),
                difficulty=float(l.get("difficulty", 0.5)),
                metadata=dict(l.get("metadata", {})),
            )

        out_edges = make_empty_adj(nodes.keys())
        in_edges = make_empty_adj(nodes.keys())

        def add_edge(src: str, dst: str) -> None:
            if src in nodes and dst in nodes:
                out_edges[src].add(dst)
                in_edges[dst].add(src)

        for l in data.get("lemmas", []):
            lid = str(l["id"])
            for dep in l.get("depends_on", []):
                add_edge(str(dep), lid)

        for dep in data.get("target_depends_on", []):
            add_edge(str(dep), target_id)

        # Optional explicit edges in trace payload.
        for src, dst in data.get("edges", []):
            add_edge(str(src), str(dst))

        graph = TAAMGraph(
            theorem_id=theorem_id,
            target_id=target_id,
            nodes=nodes,
            out_edges=out_edges,
            in_edges=in_edges,
            imports=list(data.get("imports", ["Mathlib"])),
            theorem_context=list(data.get("theorem_context", [])),
        )
        return FormalGraphExtractor.prune_syntax_nodes(graph)
