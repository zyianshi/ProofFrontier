from __future__ import annotations

import argparse
import json
from pathlib import Path

from taam.config import load_experiment_config
from taam.upstream.graph_loader import load_graph
from taam.midstream.pipeline import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="TAAM experiment pipeline")
    parser.add_argument("--config", type=str, default="", help="Path to experiment config JSON")
    parser.add_argument("--graph-json", type=str, default="", help="Path to derivation graph JSON")
    parser.add_argument("--lean-trace-json", type=str, default="", help="Path to Lean trace JSON")
    parser.add_argument("--out-dir", type=str, default="artifacts/midstream_taam_generation/manual_run", help="Output directory")
    parser.add_argument("--capability-threshold", type=float, default=0.0, help="Legacy debug field; unused by prover")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--solver-type", type=str, default="deepseek_prover", help="deepseek_prover")
    parser.add_argument("--solver-config-json", type=str, default="{}", help="Inline JSON for solver config")
    parser.add_argument("--validation-config-json", type=str, default="{}", help="Inline JSON for validation config")
    parser.add_argument(
        "--masking-strategy",
        type=str,
        default="greedy_topology",
        help="greedy_topology | random_mask | low_centrality_first",
    )
    parser.add_argument("--masking-config-json", type=str, default="{}", help="Inline JSON for masking config")
    args = parser.parse_args()

    if args.config:
        cfg = load_experiment_config(Path(args.config))
        graph_json = cfg.graph_json
        lean_trace_json = cfg.lean_trace_json
        out_dir = cfg.out_dir
        threshold = cfg.capability_threshold
        seed = cfg.seed
        solver_type = cfg.solver_type
        solver_config = cfg.solver_config
        validation_config = cfg.validation_config
        masking_strategy = cfg.masking_strategy
        masking_config = cfg.masking_config
    else:
        graph_json = args.graph_json
        lean_trace_json = args.lean_trace_json
        out_dir = args.out_dir
        threshold = args.capability_threshold
        seed = args.seed
        solver_type = args.solver_type
        solver_config = json.loads(args.solver_config_json)
        validation_config = json.loads(args.validation_config_json)
        masking_strategy = args.masking_strategy
        masking_config = json.loads(args.masking_config_json)

    graph = load_graph(graph_json=graph_json, lean_trace_json=lean_trace_json)
    sample = run_experiment(
        graph,
        Path(out_dir),
        threshold,
        seed,
        solver_type=solver_type,
        solver_config=solver_config,
        validation_config=validation_config,
        masking_strategy=masking_strategy,
        masking_config=masking_config,
    )

    if sample is None:
        print("No hard sample found: solver/prover succeeded for all masking steps.")
        return

    print("Hard sample found.")
    print(f"Theorem ID: {sample.theorem_id}")
    print(f"Target ID: {sample.target_id}")
    print(f"Failed on critical lemma: {sample.failed_on}")
    print(f"Hidden nodes: {sample.hidden_nodes}")
    print(f"Saved to: {Path(out_dir).resolve()}")


if __name__ == "__main__":
    main()
