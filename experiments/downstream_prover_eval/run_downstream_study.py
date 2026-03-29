from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taam.config import (
    DownstreamArmConfig,
    DownstreamBenchmarkRuntimeConfig,
    DownstreamBenchmarkSuiteConfig,
    load_downstream_config,
)
from taam.downstream.downstream import (
    compare_benchmark_runs,
    export_dataset_bundle,
    load_benchmark_rows,
    run_benchmark_job,
    run_training_job,
    summarize_benchmark_rows,
)
from taam.downstream.minif2f import (
    LeanBenchmarkTask,
    load_benchmark_tasks,
    load_manifest,
    prepare_benchmark_data,
    write_benchmark_manifest,
)
from taam.downstream.repair_helper import (
    is_algebra_like_task_name,
    load_jsonl_rows,
    select_best_budget_schedule,
)


def _urlopen_no_proxy(req_or_url, timeout_sec: int):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(req_or_url, timeout=timeout_sec)


def _http_get_json(url: str, timeout_sec: int) -> dict[str, Any] | None:
    try:
        with _urlopen_no_proxy(url, timeout_sec=timeout_sec) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return None


def _write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_json_if_exists(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_required_json(path: Path, label: str) -> Dict[str, Any]:
    payload = _read_json_if_exists(path)
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} did not produce a valid JSON object at {path}")
    return payload


def _prepare_helper_dataset(cfg, out_dir: Path) -> Dict[str, Any]:
    dataset_cfg = cfg.premise_helper.dataset
    manifest = export_dataset_bundle(
        source_type=dataset_cfg.source_type,
        samples_root=Path(dataset_cfg.samples_root),
        sample_glob=dataset_cfg.sample_glob,
        inventory_jsonl=Path(dataset_cfg.inventory_jsonl) if dataset_cfg.inventory_jsonl else Path(),
        out_dir=out_dir / "datasets" / "premise_helper",
        dataset_format=dataset_cfg.format,
        only_well_posed=dataset_cfg.only_well_posed,
        require_proof_completion=dataset_cfg.require_proof_completion,
        train_ratio=dataset_cfg.train_ratio,
        val_ratio=dataset_cfg.val_ratio,
        test_ratio=dataset_cfg.test_ratio,
        seed=dataset_cfg.seed,
    )
    kept = int(manifest.get("stats", {}).get("kept_samples", 0))
    if kept <= 0:
        raise SystemExit("Premise-helper dataset export produced zero samples")
    for split_name in ("train", "val", "test"):
        if int(manifest.get("split_sizes", {}).get(split_name, 0)) <= 0:
            raise SystemExit(f"Premise-helper split {split_name} is empty")
    return manifest


def _prepare_benchmark_manifest(benchmark: DownstreamBenchmarkSuiteConfig, out_dir: Path) -> Dict[str, Any]:
    if benchmark.manifest_path:
        manifest_path = Path(benchmark.manifest_path)
        tasks = load_manifest(manifest_path)
        if not tasks:
            raise SystemExit(f"Benchmark manifest is empty: {manifest_path}")
        return {"manifest_path": str(manifest_path), "data_path": benchmark.data_path, "tasks": len(tasks)}

    cache_dir = out_dir / "benchmark_sources" / benchmark.name
    resolved_data_path = prepare_benchmark_data(benchmark.data_path, benchmark.dataset_name, cache_dir)
    tasks = load_benchmark_tasks(resolved_data_path, benchmark.dataset_name, benchmark.split, benchmark.task_limit)
    if not tasks:
        raise SystemExit(f"Prepared benchmark {benchmark.name} has zero tasks from {resolved_data_path}")
    manifest_out = out_dir / "benchmark_manifests" / f"{benchmark.name}.jsonl"
    write_benchmark_manifest(tasks, manifest_out)
    return {"manifest_path": str(manifest_out), "data_path": str(resolved_data_path), "tasks": len(tasks)}


def _prepare_helper_replay_manifests(helper_manifest: Dict[str, Any], out_dir: Path) -> Dict[str, Dict[str, Any]]:
    replay_inputs: Dict[str, Dict[str, Any]] = {}
    for split_name in ("val", "test"):
        split_path = Path(helper_manifest["split_paths"][split_name])
        rows = load_jsonl_rows(split_path)
        tasks: List[LeanBenchmarkTask] = []
        for idx, row in enumerate(rows, start=1):
            query = str(row.get("query", "")).strip()
            if not query:
                continue
            theorem_name = str(row.get("theorem_id", row.get("sample_id", f"hard_sample_{idx}"))).strip() or f"hard_sample_{idx}"
            tasks.append(
                LeanBenchmarkTask(
                    task_id=str(row.get("sample_id", theorem_name)),
                    theorem_name=theorem_name,
                    split=split_name,
                    source_file=str(row.get("source_path", "")),
                    lean_code=query,
                    dataset="hard_sample_replay",
                    metadata=dict(row),
                )
            )
        manifest_path = out_dir / "benchmark_manifests" / f"hard_sample_replay_{split_name}.jsonl"
        write_benchmark_manifest(tasks, manifest_path)
        replay_inputs[split_name] = {
            "manifest_path": str(manifest_path),
            "data_path": "",
            "tasks": len(tasks),
        }
    return replay_inputs


def _arm_output_model_dir(arm: DownstreamArmConfig, arm_dir: Path) -> Path:
    if arm.output_model_dir:
        return Path(arm.output_model_dir)
    return arm_dir / "model"


def _terminate_process(proc: subprocess.Popen[str] | None, timeout_sec: int = 10) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=timeout_sec)
    except (OSError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass


def _launch_benchmark_service(
    model_ref: str,
    runtime_cfg: DownstreamBenchmarkRuntimeConfig,
    log_path: Path,
) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = str(runtime_cfg.gpu_device)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    service_log = log_path.open("a", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            [
                sys.executable,
                "tools/run_deepseek_prover.py",
                "--serve",
                "--model-id",
                model_ref,
                "--host",
                runtime_cfg.host,
                "--port",
                str(runtime_cfg.port),
                "--cuda-device",
                "0",
            ],
            cwd=ROOT,
            env=env,
            stdout=service_log,
            stderr=service_log,
            text=True,
            start_new_session=True,
        )
    finally:
        service_log.close()
    return proc


def _wait_for_service_ready(proc: subprocess.Popen[str], runtime_cfg: DownstreamBenchmarkRuntimeConfig) -> str:
    service_url = f"http://{runtime_cfg.host}:{runtime_cfg.port}"
    deadline = time.monotonic() + runtime_cfg.startup_timeout_sec
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"Resident benchmark service exited early with returncode={proc.returncode}")
        health = _http_get_json(service_url.rstrip("/") + "/health", timeout_sec=runtime_cfg.healthcheck_sec)
        if health is not None:
            if health.get("load_error"):
                raise SystemExit(f"Resident benchmark service failed to load model: {health['load_error']}")
            if health.get("loaded"):
                return service_url
        time.sleep(runtime_cfg.healthcheck_sec)
    raise SystemExit(f"Resident benchmark service did not become ready within {runtime_cfg.startup_timeout_sec}s")


def _arm_needs_resident_service(arm: DownstreamArmConfig, benchmarks: Sequence[DownstreamBenchmarkSuiteConfig]) -> bool:
    templates = [arm.benchmark_command_template] if arm.benchmark_command_template else []
    templates.extend(benchmark.command_template for benchmark in benchmarks)
    return any("{server_url}" in template for template in templates if template)


def _schedule_selection_template(arm: DownstreamArmConfig, benchmarks: Sequence[DownstreamBenchmarkSuiteConfig]) -> str:
    if arm.benchmark_command_template:
        return arm.benchmark_command_template
    for benchmark in benchmarks:
        if benchmark.command_template:
            return benchmark.command_template
    raise SystemExit(f"No benchmark command_template resolved for schedule selection arm={arm.name}")


def _choose_budget_schedule(
    cfg,
    arm: DownstreamArmConfig,
    helper_artifact_dir: str,
    base_model_ref: str,
    helper_replay_inputs: Dict[str, Dict[str, Any]],
    out_dir: Path,
) -> tuple[str, Dict[str, Dict[str, Any]]]:
    candidates = list(cfg.premise_helper.budget_schedule_candidates)
    if not candidates:
        raise SystemExit("premise_helper.budget_schedule_candidates must not be empty")
    if len(candidates) == 1:
        return candidates[0], {candidates[0]: {"tasks": 0, "successes": 0, "pass_rate": 0.0}}

    val_input = helper_replay_inputs.get("val", {})
    if int(val_input.get("tasks", 0)) <= 0:
        return candidates[0], {candidates[0]: {"tasks": 0, "successes": 0, "pass_rate": 0.0}}

    benchmark_ref = cfg.benchmarks[0]
    command_template = _schedule_selection_template(arm, cfg.benchmarks)
    schedule_scores: Dict[str, Dict[str, Any]] = {}
    service_proc: subprocess.Popen[str] | None = None
    service_url = ""
    if "{server_url}" in command_template:
        service_proc = _launch_benchmark_service(
            model_ref=base_model_ref,
            runtime_cfg=cfg.benchmark_runtime,
            log_path=out_dir / "schedule_selection" / f"{arm.name}_resident_service.log",
        )
        service_url = _wait_for_service_ready(service_proc, cfg.benchmark_runtime)
    try:
        for candidate in candidates:
            result_json = out_dir / "schedule_selection" / arm.name / f"{candidate.replace('/', '_')}.json"
            left, right = candidate.split("/", 1)
            job = run_benchmark_job(
                command_template,
                {
                    "model_ref": base_model_ref,
                    "helper_model_dir": helper_artifact_dir,
                    "premise_inventory_jsonl": cfg.premise_helper.inventory_jsonl,
                    "benchmark_name": "hard_sample_replay",
                    "split": "val",
                    "data_path": val_input["data_path"],
                    "manifest_jsonl": val_input["manifest_path"],
                    "out_json": str(result_json),
                    "timeout_sec": str(benchmark_ref.timeout_sec),
                    "arm_name": arm.name,
                    "run_name": cfg.name,
                    "server_url": service_url,
                    "pass_k": str(benchmark_ref.pass_k),
                    "temperature": str(benchmark_ref.temperature),
                    "top_p": str(benchmark_ref.top_p),
                    "max_new_tokens": str(benchmark_ref.max_new_tokens),
                    "seed": str(benchmark_ref.seed),
                    "hint_top_k": str(cfg.premise_helper.hint_top_k),
                    "budget_schedule": candidate,
                    "vanilla_attempts": left,
                    "repair_attempts": right,
                    "mode": arm.mode,
                },
                timeout_sec=benchmark_ref.timeout_sec,
            )
            if not job["success"] or not result_json.exists():
                raise SystemExit(f"Schedule selection benchmark failed for arm={arm.name}, schedule={candidate}")
            rows = load_benchmark_rows([result_json])
            schedule_scores[candidate] = summarize_benchmark_rows(rows)
    finally:
        _terminate_process(service_proc)

    return select_best_budget_schedule(candidates, schedule_scores), schedule_scores


def _run_arm_training(
    arm: DownstreamArmConfig,
    helper_manifest: Dict[str, Any],
    cfg,
    out_dir: Path,
    training_outputs: Dict[str, Dict[str, Any]],
) -> Dict[str, Any] | None:
    arm_dir = out_dir / "arms" / arm.name
    arm_dir.mkdir(parents=True, exist_ok=True)
    if arm.reuse_training_from:
        source = training_outputs.get(arm.reuse_training_from)
        if source is None:
            raise SystemExit(f"Arm {arm.name} reuses unknown training output: {arm.reuse_training_from}")
        reused = dict(source)
        reused["reused_from"] = arm.reuse_training_from
        return reused
    if not arm.training_enabled:
        return None
    if not arm.training_command_template:
        raise SystemExit(f"training_enabled=true but no training_command_template for arm={arm.name}")
    output_model_dir = _arm_output_model_dir(arm, arm_dir)
    output_model_dir.mkdir(parents=True, exist_ok=True)
    training_result_json = arm_dir / "training_result.json"
    train_result = run_training_job(
        arm.training_command_template,
        {
            "train_jsonl": helper_manifest["split_paths"]["train"],
            "eval_jsonl": helper_manifest["split_paths"]["val"],
            "test_jsonl": helper_manifest["split_paths"]["test"],
            "inventory_jsonl": cfg.premise_helper.inventory_jsonl,
            "output_model_dir": str(output_model_dir),
            "result_json": str(training_result_json),
            "base_model": arm.base_model,
            "mode": arm.mode,
            "arm_name": arm.name,
            "run_name": cfg.name,
            "timeout_sec": str(arm.training_timeout_sec),
            "helper_model_name": cfg.premise_helper.model_name,
            "bm25_candidate_count": str(cfg.premise_helper.bm25_candidate_count),
            "rerank_top_n": str(cfg.premise_helper.rerank_top_n),
            "hint_top_k": str(cfg.premise_helper.hint_top_k),
            "train_num_epochs": str(cfg.premise_helper.train_num_epochs),
            "train_batch_size": str(cfg.premise_helper.train_batch_size),
            "eval_batch_size": str(cfg.premise_helper.eval_batch_size),
            "learning_rate": str(cfg.premise_helper.learning_rate),
            "max_length": str(cfg.premise_helper.max_length),
            "seed": str(cfg.premise_helper.seed),
        },
        timeout_sec=arm.training_timeout_sec,
    )
    payload = _read_required_json(training_result_json, f"training arm={arm.name}")
    if not train_result["success"] or not payload.get("success", False):
        raise SystemExit(f"Training command failed for arm={arm.name}: returncode={train_result['returncode']}")
    if not str(payload.get("output_model_dir", "")).strip():
        raise SystemExit(f"Training result for arm={arm.name} is missing output_model_dir")
    return {
        "job": train_result,
        "result_json": str(training_result_json),
        "output_model_dir": str(payload["output_model_dir"]),
        "payload": payload,
    }


def _run_arm_benchmarks(
    arm: DownstreamArmConfig,
    benchmarks: Sequence[DownstreamBenchmarkSuiteConfig],
    benchmark_inputs: Dict[str, Dict[str, Any]],
    base_model_ref: str,
    helper_artifact_dir: str,
    budget_schedule: str,
    cfg,
    out_dir: Path,
) -> List[Dict[str, Any]]:
    arm_dir = out_dir / "arms" / arm.name
    bench_dir = arm_dir / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)

    service_proc: subprocess.Popen[str] | None = None
    service_url = ""
    if _arm_needs_resident_service(arm, benchmarks):
        service_proc = _launch_benchmark_service(
            model_ref=base_model_ref,
            runtime_cfg=cfg.benchmark_runtime,
            log_path=arm_dir / "resident_benchmark_service.log",
        )
        service_url = _wait_for_service_ready(service_proc, cfg.benchmark_runtime)

    results: List[Dict[str, Any]] = []
    vanilla_attempts = 0
    repair_attempts = 0
    if budget_schedule:
        left, right = budget_schedule.split("/", 1)
        vanilla_attempts = int(left)
        repair_attempts = int(right)
    try:
        for benchmark in benchmarks:
            command_template = arm.benchmark_command_template or benchmark.command_template
            if not command_template:
                raise SystemExit(f"No benchmark command_template resolved for arm={arm.name}, benchmark={benchmark.name}")
            benchmark_input = benchmark_inputs[benchmark.name]
            result_json = bench_dir / f"{benchmark.name}.json"
            job = run_benchmark_job(
                command_template,
                {
                    "model_ref": base_model_ref,
                    "helper_model_dir": helper_artifact_dir,
                    "premise_inventory_jsonl": cfg.premise_helper.inventory_jsonl,
                    "benchmark_name": benchmark.dataset_name,
                    "split": benchmark.split,
                    "data_path": benchmark_input["data_path"],
                    "manifest_jsonl": benchmark_input["manifest_path"],
                    "out_json": str(result_json),
                    "timeout_sec": str(benchmark.timeout_sec),
                    "arm_name": arm.name,
                    "run_name": cfg.name,
                    "server_url": service_url,
                    "pass_k": str(benchmark.pass_k),
                    "temperature": str(benchmark.temperature),
                    "top_p": str(benchmark.top_p),
                    "max_new_tokens": str(benchmark.max_new_tokens),
                    "seed": str(benchmark.seed),
                    "hint_top_k": str(cfg.premise_helper.hint_top_k),
                    "budget_schedule": budget_schedule,
                    "vanilla_attempts": str(vanilla_attempts),
                    "repair_attempts": str(repair_attempts),
                },
                timeout_sec=benchmark.timeout_sec,
            )
            if not job["success"] or not result_json.exists():
                raise SystemExit(f"Benchmark command failed for arm={arm.name}, benchmark={benchmark.name}")
            rows = load_benchmark_rows([result_json])
            results.append(
                {
                    "arm_name": arm.name,
                    "mode": arm.mode,
                    "benchmark_name": benchmark.name,
                    "dataset_name": benchmark.dataset_name,
                    "result_json": str(result_json),
                    "summary": summarize_benchmark_rows(rows),
                    "job": job,
                    "budget_schedule": budget_schedule,
                }
            )
        return results
    finally:
        _terminate_process(service_proc)


def _run_targeted_hard_replay(
    arm: DownstreamArmConfig,
    base_model_ref: str,
    helper_artifact_dir: str,
    budget_schedule: str,
    cfg,
    out_dir: Path,
    helper_replay_inputs: Dict[str, Dict[str, Any]],
) -> Dict[str, Any] | None:
    replay_input = helper_replay_inputs.get("test", {})
    if int(replay_input.get("tasks", 0)) <= 0:
        return None
    benchmark_ref = cfg.benchmarks[0]
    command_template = arm.benchmark_command_template or benchmark_ref.command_template
    if not command_template:
        return None
    result_json = out_dir / "arms" / arm.name / "targeted" / "hard_sample_replay_test.json"
    service_proc: subprocess.Popen[str] | None = None
    service_url = ""
    if "{server_url}" in command_template:
        service_proc = _launch_benchmark_service(
            model_ref=base_model_ref,
            runtime_cfg=cfg.benchmark_runtime,
            log_path=out_dir / "arms" / arm.name / "targeted" / "resident_hard_replay.log",
        )
        service_url = _wait_for_service_ready(service_proc, cfg.benchmark_runtime)
    try:
        left, right = ("0", "0")
        if budget_schedule:
            left, right = budget_schedule.split("/", 1)
        mode = arm.mode if arm.mode.startswith("repair_helper") else "vanilla"
        job = run_benchmark_job(
            command_template,
            {
                "model_ref": base_model_ref,
                "helper_model_dir": helper_artifact_dir,
                "premise_inventory_jsonl": cfg.premise_helper.inventory_jsonl,
                "benchmark_name": "hard_sample_replay",
                "split": "test",
                "data_path": replay_input["data_path"],
                "manifest_jsonl": replay_input["manifest_path"],
                "out_json": str(result_json),
                "timeout_sec": str(benchmark_ref.timeout_sec),
                "arm_name": arm.name,
                "run_name": cfg.name,
                "server_url": service_url,
                "pass_k": str(benchmark_ref.pass_k),
                "temperature": str(benchmark_ref.temperature),
                "top_p": str(benchmark_ref.top_p),
                "max_new_tokens": str(benchmark_ref.max_new_tokens),
                "seed": str(benchmark_ref.seed),
                "hint_top_k": str(cfg.premise_helper.hint_top_k),
                "budget_schedule": budget_schedule,
                "vanilla_attempts": left,
                "repair_attempts": right,
                "mode": mode,
            },
            timeout_sec=benchmark_ref.timeout_sec,
        )
        if not job["success"] or not result_json.exists():
            raise SystemExit(f"Targeted hard-sample replay failed for arm={arm.name}")
        rows = load_benchmark_rows([result_json])
        return {
            "result_json": str(result_json),
            "summary": summarize_benchmark_rows(rows),
            "rows": rows,
        }
    finally:
        _terminate_process(service_proc)


def _summarize_algebra_slice(result_json: Path) -> Dict[str, Any] | None:
    rows = load_benchmark_rows([result_json])
    filtered = [
        row
        for row in rows
        if is_algebra_like_task_name(str(row.get("theorem_name", row.get("task_id", ""))))
    ]
    if not filtered:
        return None
    return {
        "rows": filtered,
        "summary": summarize_benchmark_rows(filtered),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run downstream frozen-base repair-helper study")
    parser.add_argument("--config", type=str, default="configs/downstream_prover_eval/downstream_rl_minif2f.json")
    args = parser.parse_args()

    cfg = load_downstream_config(Path(args.config))
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    helper_manifest = _prepare_helper_dataset(cfg, out_dir)
    benchmark_inputs = {benchmark.name: _prepare_benchmark_manifest(benchmark, out_dir) for benchmark in cfg.benchmarks}
    helper_replay_inputs = _prepare_helper_replay_manifests(helper_manifest, out_dir)

    summary: Dict[str, Any] = {
        "name": cfg.name,
        "premise_helper_dataset_manifest": helper_manifest,
        "benchmark_manifests": benchmark_inputs,
        "helper_replay_manifests": helper_replay_inputs,
        "arms": [],
        "main_results": [],
        "pairwise_comparisons": [],
        "targeted_results": {
            "hard_sample_replay_test": {"by_arm": {}, "pairwise": []},
            "miniF2F_algebra_like": {"by_arm": {}, "pairwise": []},
        },
    }

    training_outputs: Dict[str, Dict[str, Any]] = {}
    arm_results_by_name: Dict[str, List[Dict[str, Any]]] = {}
    targeted_rows_by_arm: Dict[str, Dict[str, Any]] = {}

    for arm in cfg.arms:
        if not arm.enabled:
            continue
        training_output = _run_arm_training(arm, helper_manifest, cfg, out_dir, training_outputs)
        if training_output is not None:
            training_outputs[arm.name] = training_output
        helper_artifact_dir = ""
        if arm.mode.startswith("repair_helper"):
            if arm.reuse_training_from:
                helper_artifact_dir = training_outputs[arm.reuse_training_from]["output_model_dir"]
            elif training_output is not None:
                helper_artifact_dir = training_output["output_model_dir"]
            else:
                raise SystemExit(f"Repair-helper arm {arm.name} has no helper artifact source")

        base_model_ref = arm.benchmark_model_ref or arm.base_model
        budget_schedule = ""
        budget_schedule_scores: Dict[str, Dict[str, Any]] = {}
        if arm.mode == "repair_helper_budget32":
            budget_schedule, budget_schedule_scores = _choose_budget_schedule(
                cfg=cfg,
                arm=arm,
                helper_artifact_dir=helper_artifact_dir,
                base_model_ref=base_model_ref,
                helper_replay_inputs=helper_replay_inputs,
                out_dir=out_dir,
            )
        elif arm.mode == "repair_helper_extra_budget":
            budget_schedule = "32/32"
        benchmark_rows = _run_arm_benchmarks(
            arm=arm,
            benchmarks=cfg.benchmarks,
            benchmark_inputs=benchmark_inputs,
            base_model_ref=base_model_ref,
            helper_artifact_dir=helper_artifact_dir,
            budget_schedule=budget_schedule,
            cfg=cfg,
            out_dir=out_dir,
        )
        arm_results_by_name[arm.name] = benchmark_rows
        targeted_hard = _run_targeted_hard_replay(
            arm=arm,
            base_model_ref=base_model_ref,
            helper_artifact_dir=helper_artifact_dir,
            budget_schedule=budget_schedule,
            cfg=cfg,
            out_dir=out_dir,
            helper_replay_inputs=helper_replay_inputs,
        )
        algebra_slice = None
        for item in benchmark_rows:
            if item["dataset_name"] == "miniF2F":
                algebra_slice = _summarize_algebra_slice(Path(item["result_json"]))
                break
        targeted_rows_by_arm[arm.name] = {
            "hard_sample_replay_test": targeted_hard,
            "miniF2F_algebra_like": algebra_slice,
        }
        if targeted_hard is not None:
            summary["targeted_results"]["hard_sample_replay_test"]["by_arm"][arm.name] = {
                "mode": arm.mode,
                "result_json": targeted_hard["result_json"],
                **targeted_hard["summary"],
            }
        if algebra_slice is not None:
            summary["targeted_results"]["miniF2F_algebra_like"]["by_arm"][arm.name] = {
                "mode": arm.mode,
                **algebra_slice["summary"],
            }
        summary["arms"].append(
            {
                "name": arm.name,
                "mode": arm.mode,
                "training": training_output,
                "benchmarks": benchmark_rows,
                "helper_artifact_dir": helper_artifact_dir,
                "selected_budget_schedule": budget_schedule,
                "budget_schedule_scores": budget_schedule_scores,
                "targeted_evaluations": {
                    "hard_sample_replay_test": None if targeted_hard is None else targeted_hard["summary"],
                    "miniF2F_algebra_like": None if algebra_slice is None else algebra_slice["summary"],
                },
            }
        )

    main_rows: List[Dict[str, Any]] = []
    for arm_name, results in arm_results_by_name.items():
        for item in results:
            bench_summary = item["summary"]
            main_rows.append(
                {
                    "arm_name": arm_name,
                    "mode": item["mode"],
                    "benchmark_name": item["benchmark_name"],
                    "dataset_name": item["dataset_name"],
                    "tasks": bench_summary["tasks"],
                    "successes": bench_summary["successes"],
                    "pass_rate": bench_summary["pass_rate"],
                    "budget_schedule": item.get("budget_schedule", ""),
                    "result_json": item["result_json"],
                }
            )
    summary["main_results"] = main_rows

    pairwise_rows: List[Dict[str, Any]] = []
    for arm in cfg.arms:
        if not arm.enabled or not arm.compare_to:
            continue
        before = {item["benchmark_name"]: item for item in arm_results_by_name.get(arm.compare_to, [])}
        after = {item["benchmark_name"]: item for item in arm_results_by_name.get(arm.name, [])}
        for benchmark in cfg.benchmarks:
            before_item = before.get(benchmark.name)
            after_item = after.get(benchmark.name)
            if before_item is None or after_item is None:
                continue
            comparison = compare_benchmark_runs(
                load_benchmark_rows([Path(before_item["result_json"])]),
                load_benchmark_rows([Path(after_item["result_json"])]),
            )
            pairwise_rows.append(
                {
                    "arm_name": arm.name,
                    "compare_to": arm.compare_to,
                    "benchmark_name": benchmark.name,
                    "dataset_name": benchmark.dataset_name,
                    "matched_tasks": comparison["matched_tasks"],
                    "before_pass_rate": comparison["before"]["pass_rate"],
                    "after_pass_rate": comparison["after"]["pass_rate"],
                    "absolute_gain": comparison["absolute_gain"],
                    "error_reduction": comparison["error_reduction"],
                    "improved_tasks": comparison["improved_tasks"],
                    "regressed_tasks": comparison["regressed_tasks"],
                    "win_rate": comparison["win_rate"],
                    "regression_rate": comparison["regression_rate"],
                }
            )
    summary["pairwise_comparisons"] = pairwise_rows

    for arm in cfg.arms:
        if not arm.enabled or not arm.compare_to:
            continue
        before_targeted = targeted_rows_by_arm.get(arm.compare_to, {})
        after_targeted = targeted_rows_by_arm.get(arm.name, {})
        before_hard = before_targeted.get("hard_sample_replay_test")
        after_hard = after_targeted.get("hard_sample_replay_test")
        if before_hard and after_hard:
            comparison = compare_benchmark_runs(before_hard["rows"], after_hard["rows"])
            summary["targeted_results"]["hard_sample_replay_test"]["pairwise"].append(
                {
                    "arm_name": arm.name,
                    "compare_to": arm.compare_to,
                    "matched_tasks": comparison["matched_tasks"],
                    "before_pass_rate": comparison["before"]["pass_rate"],
                    "after_pass_rate": comparison["after"]["pass_rate"],
                    "absolute_gain": comparison["absolute_gain"],
                }
            )
        before_algebra = before_targeted.get("miniF2F_algebra_like")
        after_algebra = after_targeted.get("miniF2F_algebra_like")
        if before_algebra and after_algebra:
            comparison = compare_benchmark_runs(before_algebra["rows"], after_algebra["rows"])
            summary["targeted_results"]["miniF2F_algebra_like"]["pairwise"].append(
                {
                    "arm_name": arm.name,
                    "compare_to": arm.compare_to,
                    "matched_tasks": comparison["matched_tasks"],
                    "before_pass_rate": comparison["before"]["pass_rate"],
                    "after_pass_rate": comparison["after"]["pass_rate"],
                    "absolute_gain": comparison["absolute_gain"],
                }
            )

    summary_path = out_dir / "downstream_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(main_rows, out_dir / "main_results.csv")
    _write_csv(pairwise_rows, out_dir / "pairwise_comparisons.csv")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
