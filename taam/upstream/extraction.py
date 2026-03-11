from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from ..types import GraphNode, NodeId, TAAMGraph, make_empty_adj


class FormalGraphExtractor:
    """
    Phase 1:
    Load/construct a derivation DAG and prune low-value syntax-only nodes.
    """

    PRUNE_KEYWORDS = {"cast", "coe", "rfl", "simp", "trivial"}

    @staticmethod
    def from_json(path: Path) -> TAAMGraph:
        data = json.loads(path.read_text(encoding="utf-8"))
        nodes = {
            n["id"]: GraphNode(
                node_id=n["id"],
                kind=n["kind"],
                statement=n["statement"],
                lean_statement=str(n.get("lean_statement", n.get("statement", ""))),
                difficulty=float(n.get("difficulty", 0.5)),
                metadata=dict(n.get("metadata", {})),
            )
            for n in data["nodes"]
        }

        out_edges = make_empty_adj(nodes.keys())
        in_edges = make_empty_adj(nodes.keys())
        for src, dst in data["edges"]:
            if src not in nodes or dst not in nodes:
                continue
            out_edges[src].add(dst)
            in_edges[dst].add(src)

        graph = TAAMGraph(
            theorem_id=data.get("theorem_id", "unknown_theorem"),
            target_id=data["target_id"],
            nodes=nodes,
            out_edges=out_edges,
            in_edges=in_edges,
            imports=list(data.get("imports", ["Mathlib"])),
            theorem_context=list(data.get("theorem_context", [])),
        )
        return FormalGraphExtractor.prune_syntax_nodes(graph)

    @staticmethod
    def prune_syntax_nodes(graph: TAAMGraph) -> TAAMGraph:
        keep: Dict[NodeId, GraphNode] = {}
        for nid, node in graph.nodes.items():
            if node.kind in {"premise", "target"}:
                keep[nid] = node
                continue
            text = node.formal_statement().lower()
            if any(k in text for k in FormalGraphExtractor.PRUNE_KEYWORDS):
                continue
            keep[nid] = node

        out_edges = make_empty_adj(keep.keys())
        in_edges = make_empty_adj(keep.keys())
        for src, dsts in graph.out_edges.items():
            if src not in keep:
                continue
            for dst in dsts:
                if dst not in keep:
                    continue
                out_edges[src].add(dst)
                in_edges[dst].add(src)

        return TAAMGraph(
            theorem_id=graph.theorem_id,
            target_id=graph.target_id,
            nodes=keep,
            out_edges=out_edges,
            in_edges=in_edges,
            imports=list(graph.imports),
            theorem_context=list(graph.theorem_context),
        )
