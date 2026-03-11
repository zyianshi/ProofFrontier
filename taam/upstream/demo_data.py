from __future__ import annotations

from ..types import GraphNode, TAAMGraph, make_empty_adj


def build_demo_graph() -> TAAMGraph:
    nodes = {
        "P1": GraphNode("P1", "premise", "a > b", "a > b", 0.2),
        "P2": GraphNode("P2", "premise", "b > 0", "b > 0", 0.2),
        "L1": GraphNode("L1", "lemma", "a - b > 0", "a - b > 0", 0.5),
        "L2": GraphNode("L2", "lemma", "a^2 - b^2 = (a-b)(a+b)", "a ^ 2 - b ^ 2 = (a - b) * (a + b)", 0.7),
        "L3": GraphNode("L3", "lemma", "a + b > 0", "a + b > 0", 0.6),
        "T": GraphNode("T", "target", "a^2 - b^2 > 0", "a ^ 2 - b ^ 2 > 0", 0.9),
    }
    edges = [("P1", "L2"), ("P2", "L1"), ("P2", "L3"), ("L1", "T"), ("L2", "T"), ("L3", "T")]
    out_edges = make_empty_adj(nodes.keys())
    in_edges = make_empty_adj(nodes.keys())
    for src, dst in edges:
        out_edges[src].add(dst)
        in_edges[dst].add(src)
    return TAAMGraph(
        "demo_theorem",
        "T",
        nodes,
        out_edges,
        in_edges,
        imports=["Mathlib"],
        theorem_context=["{a b : Real}"],
    )
