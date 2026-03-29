from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from ..external_tools import run_command_template
from .repair_helper import inject_premise_hints


@dataclass
class RepairBenchmarkTask:
    task_id: str
    theorem_name: str
    split: str
    source_file: str
    lean_code: str
    dataset: str
    language: str = "lean4"
    metadata: Dict[str, Any] = field(default_factory=dict)


def partition_repair_tasks(tasks: Sequence[RepairBenchmarkTask], shard_count: int) -> List[List[RepairBenchmarkTask]]:
    if shard_count <= 0:
        raise ValueError(f"shard_count must be positive, got {shard_count}")
    shards: List[List[RepairBenchmarkTask]] = [[] for _ in range(shard_count)]
    for index, task in enumerate(tasks):
        shards[index % shard_count].append(task)
    return shards


def merge_repair_benchmark_payloads(
    payloads: Sequence[Dict[str, Any]],
    *,
    ordered_task_ids: Sequence[str],
) -> Dict[str, Any]:
    if not payloads:
        raise ValueError("payloads must be non-empty")
    first = payloads[0]
    merged_results_by_id: Dict[str, Dict[str, Any]] = {}
    for payload in payloads:
        for key in ("benchmark_name", "split", "pass_k", "mode", "budget_schedule", "hint_source"):
            if payload.get(key) != first.get(key):
                raise ValueError(f"Mismatched payload field {key}: {payload.get(key)!r} != {first.get(key)!r}")
        for row in payload.get("results", []):
            task_id = str(row.get("task_id", ""))
            if not task_id:
                raise ValueError("Encountered repair benchmark row without task_id")
            merged_results_by_id[task_id] = row

    ordered_results = [merged_results_by_id[task_id] for task_id in ordered_task_ids if task_id in merged_results_by_id]
    successes = sum(1 for row in ordered_results if bool(row.get("success")))
    return {
        "benchmark_name": first.get("benchmark_name", ""),
        "split": first.get("split", ""),
        "tasks": len(ordered_results),
        "successes": successes,
        "pass_k": int(first.get("pass_k", 0)),
        "mode": first.get("mode", ""),
        "budget_schedule": first.get("budget_schedule", ""),
        "hint_source": first.get("hint_source", ""),
        "results": ordered_results,
    }


def allocate_repair_attempt_budgets(mode: str, pass_k: int, schedule_name: str) -> Dict[str, int]:
    if mode == "repair_helper_extra_budget":
        return {"vanilla_attempts": pass_k, "repair_attempts": pass_k}
    left, right = schedule_name.split("/", 1)
    vanilla_attempts = int(left)
    repair_attempts = int(right)
    if vanilla_attempts + repair_attempts != pass_k:
        raise ValueError(f"Budget schedule {schedule_name} does not sum to pass_k={pass_k}")
    return {"vanilla_attempts": vanilla_attempts, "repair_attempts": repair_attempts}


def _run_attempt(
    *,
    lean_text: str,
    theorem_name: str,
    out_dir: Path,
    command_template: str,
    timeout_sec: int,
    model_ref: str,
    server_url: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    seed: int,
    stage: str,
    attempt_index: int,
) -> Dict[str, Any]:
    lean_file = out_dir / f"{theorem_name}_{stage}_{attempt_index:02d}.lean"
    result_json = out_dir / f"{theorem_name}_{stage}_{attempt_index:02d}.json"
    lean_file.write_text(lean_text, encoding="utf-8")
    try:
        proc = run_command_template(
            command_template,
            {
                "lean_file": str(lean_file),
                "result_json": str(result_json),
                "out_json": str(result_json),
                "timeout_sec": str(timeout_sec),
                "model_ref": model_ref,
                "server_url": server_url,
                "temperature": str(temperature),
                "top_p": str(top_p),
                "max_new_tokens": str(max_new_tokens),
                "seed": str(seed),
            },
            timeout_sec=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "stage": stage, "attempt_index": attempt_index, "error": "timeout"}
    payload = {}
    if result_json.exists():
        payload = json.loads(result_json.read_text(encoding="utf-8"))
    solved = bool(payload.get("solved", payload.get("success", proc.returncode == 0)))
    return {
        "success": solved,
        "stage": stage,
        "attempt_index": attempt_index,
        "returncode": proc.returncode,
        "result_json": str(result_json),
    }


def run_repair_task(
    *,
    task: RepairBenchmarkTask,
    command_template: str,
    model_ref: str,
    server_url: str,
    timeout_sec: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    seed: int,
    vanilla_attempts: int,
    repair_attempts: int,
    helper_hits: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="taam_repair_benchmark_") as tmp_dir:
        tmp = Path(tmp_dir)
        vanilla_results: List[Dict[str, Any]] = []
        repair_results: List[Dict[str, Any]] = []
        for attempt_index in range(vanilla_attempts):
            attempt = _run_attempt(
                lean_text=task.lean_code,
                theorem_name=task.theorem_name,
                out_dir=tmp,
                command_template=command_template,
                timeout_sec=timeout_sec,
                model_ref=model_ref,
                server_url=server_url,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                seed=seed + attempt_index,
                stage="vanilla",
                attempt_index=attempt_index,
            )
            vanilla_results.append(attempt)
            if attempt["success"]:
                return {
                    "task_id": task.task_id,
                    "success": True,
                    "successful_stage": "vanilla",
                    "vanilla_success": True,
                    "repair_success": False,
                    "vanilla_attempt_results": vanilla_results,
                    "repair_attempt_results": [],
                }

        hinted_problem = inject_premise_hints(task.lean_code, helper_hits)
        for attempt_index in range(repair_attempts):
            attempt = _run_attempt(
                lean_text=hinted_problem,
                theorem_name=task.theorem_name,
                out_dir=tmp,
                command_template=command_template,
                timeout_sec=timeout_sec,
                model_ref=model_ref,
                server_url=server_url,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                seed=seed + vanilla_attempts + attempt_index,
                stage="repair",
                attempt_index=attempt_index,
            )
            repair_results.append(attempt)
            if attempt["success"]:
                return {
                    "task_id": task.task_id,
                    "success": True,
                    "successful_stage": "repair",
                    "vanilla_success": False,
                    "repair_success": True,
                    "vanilla_attempt_results": vanilla_results,
                    "repair_attempt_results": repair_results,
                }

        return {
            "task_id": task.task_id,
            "success": False,
            "successful_stage": "",
            "vanilla_success": False,
            "repair_success": False,
            "vanilla_attempt_results": vanilla_results,
            "repair_attempt_results": repair_results,
        }


def write_repair_manifest(tasks: Iterable[RepairBenchmarkTask], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(asdict(task), ensure_ascii=False) + "\n")
