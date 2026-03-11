from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taam.config import load_experiment_config
from taam.upstream.graph_loader import load_graph
from taam.midstream.pipeline import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one TAAM experiment from config")
    parser.add_argument("--config", type=str, default="configs/midstream_taam_generation/default_experiment.json")
    args = parser.parse_args()

    cfg = load_experiment_config(Path(args.config))
    graph = load_graph(graph_json=cfg.graph_json, lean_trace_json=cfg.lean_trace_json)
    sample = run_experiment(
        graph,
        Path(cfg.out_dir),
        cfg.capability_threshold,
        cfg.seed,
        solver_type=cfg.solver_type,
        solver_config=cfg.solver_config,
        validation_config=cfg.validation_config,
        masking_strategy=cfg.masking_strategy,
        masking_config=cfg.masking_config,
    )

    if sample is None:
        print(f"[{cfg.name}] No hard sample found.")
        return

    print(f"[{cfg.name}] Hard sample found.")
    print(f"theorem={sample.theorem_id}, target={sample.target_id}, failed_on={sample.failed_on}")
    print(f"saved_dir={Path(cfg.out_dir).resolve()}")


if __name__ == "__main__":
    main()
