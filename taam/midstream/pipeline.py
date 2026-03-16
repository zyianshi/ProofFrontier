from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from .formalization import build_lean4_problem_from_state
from .masking import TAAMMasker
from .profiling import TopologicalProfiler
from .solver import create_solver
from .validation import SafetyNetValidator
from ..types import HardProblemSample, ProblemState, TAAMGraph


def _edge_count(graph: TAAMGraph) -> int:
    return sum(len(v) for v in graph.out_edges.values())


def _infer_theorem_domain(theorem_id: str, source_file_path: str) -> str:
    file_path = source_file_path.replace("\\", "/").strip()
    if file_path:
        if file_path.endswith('.lean'):
            file_path = file_path[:-5]
        return file_path.replace('/', '.')

    parts = [p for p in theorem_id.split('.') if p]
    if not parts:
        return ''
    if len(parts) == 1:
        return parts[0]
    if parts[0] == 'Mathlib' and len(parts) >= 3:
        return '.'.join(parts[:-1])
    return '.'.join(parts[:-1])


def save_sample(sample: HardProblemSample, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_path = out_dir / f"{sample.theorem_id}_{sample.target_id}_hard_sample.json"
    sample.sample_json_path = str(sample_path.resolve())
    sample_path.write_text(json.dumps(asdict(sample), ensure_ascii=False, indent=2), encoding='utf-8')
    return sample_path


def run_experiment(
    graph: TAAMGraph,
    out_dir: Path,
    threshold: float,
    seed: int,
    solver_type: str = 'deepseek_prover',
    solver_config: Optional[Dict[str, Any]] = None,
    validation_config: Optional[Dict[str, Any]] = None,
    masking_strategy: str = 'greedy_topology',
    masking_config: Optional[Dict[str, Any]] = None,
) -> Optional[HardProblemSample]:
    mcfg = masking_config or {}
    masking_seed = int(mcfg.get('seed', seed))
    ranked = TopologicalProfiler.rank_candidates(graph)
    solver = create_solver(solver_type, threshold, seed, solver_config=solver_config)
    masker = TAAMMasker(solver, masking_strategy=masking_strategy, seed=masking_seed)
    sample = masker.run(graph, ranked)
    if sample is None:
        return None

    formal_state = ProblemState(
        theorem_id=sample.theorem_id,
        target_id=sample.target_id,
        target_statement=graph.nodes[sample.target_id].formal_statement(),
        visible_node_ids=sample.visible_nodes,
        hidden_node_ids=sample.hidden_nodes,
        visible_statements=[graph.nodes[n].formal_statement() for n in sample.visible_nodes],
        imports=list(graph.imports),
        theorem_context=list(graph.theorem_context),
    )
    full_visible = sorted(set(graph.premises()) | set(graph.lemmas()))
    full_state = ProblemState(
        theorem_id=sample.theorem_id,
        target_id=sample.target_id,
        target_statement=graph.nodes[sample.target_id].formal_statement(),
        visible_node_ids=full_visible,
        hidden_node_ids=[],
        visible_statements=[graph.nodes[n].formal_statement() for n in full_visible],
        imports=list(graph.imports),
        theorem_context=list(graph.theorem_context),
    )
    sample.lean_problem = build_lean4_problem_from_state(formal_state)
    sample.full_lean_problem = build_lean4_problem_from_state(full_state)

    target_meta = graph.nodes[sample.target_id].metadata
    sample.proof_completion = str(
        target_meta.get('proof_completion', target_meta.get('proof', target_meta.get('tactic_script', '')))
    ).strip()
    sample.proof_source = str(target_meta.get('proof_source', target_meta.get('proof_origin', ''))).strip()
    sample.source_file_path = str(target_meta.get('file_path', target_meta.get('source_file_path', ''))).strip()
    sample.theorem_domain = _infer_theorem_domain(sample.theorem_id, sample.source_file_path)
    sample.source_graph_size = len(graph.nodes)
    sample.source_graph_edge_count = _edge_count(graph)
    sample.hidden_lemma_count = len(sample.hidden_nodes)
    sample.visible_lemma_count = sum(1 for node_id in sample.visible_nodes if graph.nodes[node_id].kind == 'lemma')
    sample.well_posed = SafetyNetValidator.validate(sample, graph, validation_config=validation_config)
    save_sample(sample, out_dir)
    return sample
