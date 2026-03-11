from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple


NodeId = str


def make_empty_adj(nodes: Iterable[NodeId]) -> Dict[NodeId, Set[NodeId]]:
    return {nid: set() for nid in nodes}


@dataclass
class GraphNode:
    node_id: NodeId
    kind: str  # premise | lemma | target
    statement: str
    lean_statement: str = ""
    difficulty: float = 0.5
    metadata: Dict[str, str] = field(default_factory=dict)

    def formal_statement(self) -> str:
        return self.lean_statement if self.lean_statement else self.statement


@dataclass
class TAAMGraph:
    theorem_id: str
    target_id: NodeId
    nodes: Dict[NodeId, GraphNode]
    out_edges: Dict[NodeId, Set[NodeId]]
    in_edges: Dict[NodeId, Set[NodeId]]
    imports: List[str] = field(default_factory=lambda: ["Mathlib"])
    theorem_context: List[str] = field(default_factory=list)

    def premises(self) -> List[NodeId]:
        return [n.node_id for n in self.nodes.values() if n.kind == "premise"]

    def lemmas(self) -> List[NodeId]:
        return [n.node_id for n in self.nodes.values() if n.kind == "lemma"]

    def target(self) -> NodeId:
        return self.target_id

    def copy(self) -> "TAAMGraph":
        return TAAMGraph(
            theorem_id=self.theorem_id,
            target_id=self.target_id,
            nodes={k: GraphNode(**asdict(v)) for k, v in self.nodes.items()},
            out_edges={k: set(v) for k, v in self.out_edges.items()},
            in_edges={k: set(v) for k, v in self.in_edges.items()},
            imports=list(self.imports),
            theorem_context=list(self.theorem_context),
        )


@dataclass
class ProblemState:
    theorem_id: str
    target_id: str
    target_statement: str
    visible_node_ids: List[str]
    hidden_node_ids: List[str]
    visible_statements: List[str]
    imports: List[str] = field(default_factory=lambda: ["Mathlib"])
    theorem_context: List[str] = field(default_factory=list)


@dataclass
class HardProblemSample:
    theorem_id: str
    target_id: str
    failed_on: str
    hidden_nodes: List[str]
    visible_nodes: List[str]
    candidate_ranking: List[Tuple[str, float]]
    masking_strategy: str = "greedy_topology"
    lean_problem: str = ""
    full_lean_problem: str = ""
    proof_completion: str = ""
    proof_source: str = ""
    well_posed: Optional[bool] = None
