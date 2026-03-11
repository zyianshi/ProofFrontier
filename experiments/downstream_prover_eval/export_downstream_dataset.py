from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taam.config import load_downstream_config
from taam.downstream.downstream import export_dataset_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Export TAAM hard samples into downstream training datasets")
    parser.add_argument("--config", type=str, default="", help="Downstream config JSON")
    parser.add_argument("--samples-root", type=str, default="", help="Folder containing hard sample JSON files")
    parser.add_argument("--sample-glob", type=str, default="**/*_hard_sample.json")
    parser.add_argument("--out-dir", type=str, default="artifacts/downstream_prover_eval/export_bundle")
    parser.add_argument("--format", type=str, default="rl", help="rl | sft")
    parser.add_argument("--only-well-posed", action="store_true", default=False)
    parser.add_argument("--require-proof-completion", action="store_true", default=False)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.config:
        cfg = load_downstream_config(Path(args.config))
        dataset_cfg = cfg.dataset
        samples_root = Path(dataset_cfg.samples_root)
        sample_glob = dataset_cfg.sample_glob
        out_dir = Path(cfg.out_dir) / "dataset"
        dataset_format = dataset_cfg.format
        only_well_posed = dataset_cfg.only_well_posed
        require_proof_completion = dataset_cfg.require_proof_completion
        train_ratio = dataset_cfg.train_ratio
        val_ratio = dataset_cfg.val_ratio
        test_ratio = dataset_cfg.test_ratio
        seed = dataset_cfg.seed
    else:
        samples_root = Path(args.samples_root or "artifacts/midstream_taam_generation/runs")
        sample_glob = args.sample_glob
        out_dir = Path(args.out_dir)
        dataset_format = args.format
        only_well_posed = bool(args.only_well_posed)
        require_proof_completion = bool(args.require_proof_completion)
        train_ratio = args.train_ratio
        val_ratio = args.val_ratio
        test_ratio = args.test_ratio
        seed = args.seed

    manifest = export_dataset_bundle(
        samples_root=samples_root,
        sample_glob=sample_glob,
        out_dir=out_dir,
        dataset_format=dataset_format,
        only_well_posed=only_well_posed,
        require_proof_completion=require_proof_completion,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
