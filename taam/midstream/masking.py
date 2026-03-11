from __future__ import annotations

import random
from typing import List, Optional, Set, Tuple

from .solver import Solver
from ..types import HardProblemSample, NodeId, ProblemState, TAAMGraph


class TAAMMasker:
    """
    Phase 3:
    Greedy topology-guided adversarial masking loop.
    """

    def __init__(self, solver: Solver, masking_strategy: str = "greedy_topology", seed: int = 7):
        self.solver = solver
        self.masking_strategy = masking_strategy
        self._rnd = random.Random(seed)

    def run(self, graph: TAAMGraph, ranked_candidates: List[Tuple[NodeId, float]]) -> Optional[HardProblemSample]:
        visible = set(graph.premises()) | set(graph.lemmas())
        hidden: Set[NodeId] = set()
        target = graph.target()
        candidates = self._order_candidates(ranked_candidates)

        for node_id, _score in candidates:
            if node_id not in visible:
                continue
            visible.remove(node_id)
            hidden.add(node_id)
            state = self._build_problem_state(graph, visible, hidden)
            solved = self.solver.solve(state, graph)
            if not solved:
                return HardProblemSample(
                    theorem_id=graph.theorem_id,
                    target_id=target,
                    failed_on=node_id,
                    hidden_nodes=sorted(hidden),
                    visible_nodes=sorted(visible),
                    candidate_ranking=candidates,
                    masking_strategy=self.masking_strategy,
                )
        return None

    def _order_candidates(self, ranked_candidates: List[Tuple[NodeId, float]]) -> List[Tuple[NodeId, float]]:
        strategy = self.masking_strategy.lower().strip()
        if strategy == "greedy_topology":
            return list(ranked_candidates)
        if strategy == "low_centrality_first":
            return sorted(ranked_candidates, key=lambda x: x[1])
        if strategy == "random_mask":
            arr = list(ranked_candidates)
            self._rnd.shuffle(arr)
            return arr
        raise ValueError(f"Unknown masking_strategy: {self.masking_strategy}")

    @staticmethod
    def _build_problem_state(graph: TAAMGraph, visible: Set[NodeId], hidden: Set[NodeId]) -> ProblemState:
        visible_ids = sorted(visible)
        return ProblemState(
            theorem_id=graph.theorem_id,
            target_id=graph.target(),
            target_statement=graph.nodes[graph.target()].formal_statement(),
            visible_node_ids=visible_ids,
            hidden_node_ids=sorted(hidden),
            visible_statements=[graph.nodes[n].formal_statement() for n in visible_ids],
            imports=list(graph.imports),
            theorem_context=list(graph.theorem_context),
        )
