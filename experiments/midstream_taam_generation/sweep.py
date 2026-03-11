from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taam.config import load_sweep_config
from taam.upstream.graph_loader import load_graph
from taam.midstream.pipeline import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TAAM hyper-parameter sweep")
    parser.add_argument("--config", type=str, default="configs/midstream_taam_generation/sweep_small.json")
    args = parser.parse_args()

    cfg = load_sweep_config(Path(args.config))
    graph = load_graph(graph_json=cfg.graph_json, lean_trace_json=cfg.lean_trace_json)

    out_root = Path(cfg.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / f"{cfg.name}_summary.csv"

    rows = []
    for threshold in cfg.capability_thresholds:
        for seed in cfg.seeds:
            out_dir = out_root / f"thr_{threshold:.2f}_seed_{seed}"
            sample = run_experiment(
                graph.copy(),
                out_dir,
                threshold,
                seed,
                solver_type=cfg.solver_type,
                solver_config=cfg.solver_config,
                validation_config=cfg.validation_config,
                masking_strategy=cfg.masking_strategy,
                masking_config=cfg.masking_config,
            )
            rows.append(
                {
                    "sweep_name": cfg.name,
                    "threshold": threshold,
                    "seed": seed,
                    "solver_type": cfg.solver_type,
                    "masking_strategy": cfg.masking_strategy,
                    "found_hard_sample": sample is not None,
                    "failed_on": "" if sample is None else sample.failed_on,
                    "hidden_count": 0 if sample is None else len(sample.hidden_nodes),
                    "well_posed": "" if sample is None else sample.well_posed,
                    "theorem_id": "" if sample is None else sample.theorem_id,
                    "target_id": "" if sample is None else sample.target_id,
                    "out_dir": str(out_dir),
                }
            )
            print(
                f"threshold={threshold:.2f}, seed={seed}, found={sample is not None}, "
                f"failed_on={'' if sample is None else sample.failed_on}"
            )

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sweep_name",
                "threshold",
                "seed",
                "solver_type",
                "masking_strategy",
                "found_hard_sample",
                "failed_on",
                "hidden_count",
                "well_posed",
                "theorem_id",
                "target_id",
                "out_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
