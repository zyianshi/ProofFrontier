from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taam.config import load_downstream_config
from taam.downstream.downstream import (
    compare_benchmark_runs,
    export_dataset_bundle,
    load_benchmark_rows,
    run_benchmark_job,
    run_training_job,
)
from taam.downstream.minif2f import load_miniF2F_tasks, write_miniF2F_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run downstream TAAM study: export dataset, train prover, compare benchmarks")
    parser.add_argument("--config", type=str, default="configs/downstream_prover_eval/downstream_rl_minif2f.json")
    args = parser.parse_args()

    cfg = load_downstream_config(Path(args.config))
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = out_dir / "dataset"
    manifest = export_dataset_bundle(
        samples_root=Path(cfg.dataset.samples_root),
        sample_glob=cfg.dataset.sample_glob,
        out_dir=dataset_dir,
        dataset_format=cfg.dataset.format,
        only_well_posed=cfg.dataset.only_well_posed,
        require_proof_completion=cfg.dataset.require_proof_completion,
        train_ratio=cfg.dataset.train_ratio,
        val_ratio=cfg.dataset.val_ratio,
        test_ratio=cfg.dataset.test_ratio,
        seed=cfg.dataset.seed,
    )

    summary: dict = {
        "name": cfg.name,
        "dataset_manifest": manifest,
        "training": None,
        "benchmark": None,
    }

    trained_model_ref = ""
    if cfg.training.enabled:
        if not cfg.training.command_template:
            raise SystemExit("training.enabled=true but training.command_template is empty")
        output_model_dir = Path(cfg.training.output_model_dir)
        output_model_dir.mkdir(parents=True, exist_ok=True)
        training_result_json = out_dir / "training_result.json"
        train_result = run_training_job(
            cfg.training.command_template,
            {
                "train_jsonl": manifest["split_paths"]["train"],
                "eval_jsonl": manifest["split_paths"]["val"],
                "test_jsonl": manifest["split_paths"]["test"],
                "output_model_dir": str(output_model_dir),
                "result_json": str(training_result_json),
                "base_model": cfg.training.base_model,
                "mode": cfg.training.mode,
                "timeout_sec": str(cfg.training.timeout_sec),
            },
            timeout_sec=cfg.training.timeout_sec,
        )
        summary["training"] = train_result
        trained_model_ref = str(output_model_dir)
    else:
        trained_model_ref = cfg.benchmark.tuned_model_ref or cfg.training.output_model_dir

    if cfg.benchmark.enabled:
        if not cfg.benchmark.base_command_template or not cfg.benchmark.tuned_command_template:
            raise SystemExit("benchmark.enabled=true but benchmark command templates are empty")
        base_result_path = out_dir / "benchmark_before.json"
        tuned_result_path = out_dir / "benchmark_after.json"
        manifest_path = ""
        if cfg.benchmark.dataset_name.lower() == "minif2f":
            if cfg.benchmark.manifest_path:
                manifest_path = cfg.benchmark.manifest_path
            elif cfg.benchmark.data_path:
                manifest_out = out_dir / f"miniF2F_{cfg.benchmark.split}_manifest.jsonl"
                tasks = load_miniF2F_tasks(
                    Path(cfg.benchmark.data_path),
                    split=cfg.benchmark.split,
                    task_limit=cfg.benchmark.task_limit,
                )
                write_miniF2F_manifest(tasks, manifest_out)
                manifest_path = str(manifest_out)

        base_job = run_benchmark_job(
            cfg.benchmark.base_command_template,
            {
                "model_ref": cfg.benchmark.base_model_ref,
                "benchmark_name": cfg.benchmark.dataset_name,
                "split": cfg.benchmark.split,
                "data_path": cfg.benchmark.data_path,
                "manifest_jsonl": manifest_path,
                "out_json": str(base_result_path),
                "timeout_sec": str(cfg.benchmark.timeout_sec),
            },
            timeout_sec=cfg.benchmark.timeout_sec,
        )
        tuned_job = run_benchmark_job(
            cfg.benchmark.tuned_command_template,
            {
                "model_ref": trained_model_ref,
                "benchmark_name": cfg.benchmark.dataset_name,
                "split": cfg.benchmark.split,
                "data_path": cfg.benchmark.data_path,
                "manifest_jsonl": manifest_path,
                "out_json": str(tuned_result_path),
                "timeout_sec": str(cfg.benchmark.timeout_sec),
            },
            timeout_sec=cfg.benchmark.timeout_sec,
        )

        benchmark_summary = {
            "base_job": base_job,
            "tuned_job": tuned_job,
        }
        if base_job["success"] and tuned_job["success"] and base_result_path.exists() and tuned_result_path.exists():
            before_rows = load_benchmark_rows([base_result_path])
            after_rows = load_benchmark_rows([tuned_result_path])
            benchmark_summary["comparison"] = compare_benchmark_runs(before_rows, after_rows)
        else:
            benchmark_summary["comparison"] = None
        summary["benchmark"] = benchmark_summary

    summary_path = out_dir / "downstream_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
