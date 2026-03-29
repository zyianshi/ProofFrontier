from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taam.downstream.minif2f import load_benchmark_tasks, load_manifest, write_benchmark_manifest
from taam.external_tools import parse_json_result, parse_verdict_from_text, run_command_template


def _load_tasks(args: argparse.Namespace):
    if args.manifest_jsonl:
        return load_manifest(Path(args.manifest_jsonl))
    if not args.data_path:
        raise SystemExit("Provide --manifest-jsonl or --data-path for theorem benchmark")
    tasks = load_benchmark_tasks(
        Path(args.data_path),
        dataset_name=args.benchmark_name,
        split=args.split,
        task_limit=args.task_limit,
    )
    if args.write_manifest:
        write_benchmark_manifest(tasks, Path(args.write_manifest))
    return tasks


def _run_attempt(
    task,
    tmp_dir: Path,
    command_template: str,
    timeout_sec: int,
    model_ref: str,
    benchmark_name: str,
    server_url: str,
    pass_k: int,
    attempt_index: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    seed: int,
) -> Dict[str, Any]:
    lean_file = tmp_dir / f"{task.theorem_name}.lean"
    result_json = tmp_dir / f"result_{attempt_index:02d}.json"
    lean_file.write_text(task.lean_code, encoding="utf-8")

    try:
        proc = run_command_template(
            command_template,
            {
                "lean_file": str(lean_file),
                "result_json": str(result_json),
                "out_json": str(result_json),
                "timeout_sec": str(timeout_sec),
                "model_ref": model_ref,
                "theorem_name": task.theorem_name,
                "task_id": task.task_id,
                "benchmark_name": benchmark_name,
                "split": task.split,
                "server_url": server_url,
                "pass_k": str(pass_k),
                "attempt_index": str(attempt_index),
                "temperature": str(temperature),
                "top_p": str(top_p),
                "max_new_tokens": str(max_new_tokens),
                "seed": str(seed),
            },
            timeout_sec=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return {
            "attempt_index": attempt_index,
            "seed": seed,
            "success": False,
            "error": "timeout",
        }

    json_verdict = parse_json_result(result_json, ("solved", "proved", "success", "passed"))
    if json_verdict is None:
        json_verdict = parse_verdict_from_text(f"{proc.stdout}\n{proc.stderr}", "TAAM_PROVER_VERDICT")
    success = proc.returncode == 0 if json_verdict is None else json_verdict
    return {
        "attempt_index": attempt_index,
        "seed": seed,
        "success": bool(success),
        "returncode": proc.returncode,
    }


def _run_task(task, args: argparse.Namespace, task_index: int) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="taam_benchmark_") as tmp_dir:
        tmp = Path(tmp_dir)
        attempt_results: List[Dict[str, Any]] = []
        task_success = False
        successful_attempt = -1
        for attempt_index in range(args.pass_k):
            attempt_seed = args.seed + ((task_index - 1) * args.pass_k) + attempt_index
            attempt = _run_attempt(
                task,
                tmp,
                args.command_template,
                timeout_sec=args.timeout_sec,
                model_ref=args.model_ref,
                benchmark_name=args.benchmark_name,
                server_url=args.server_url,
                pass_k=args.pass_k,
                attempt_index=attempt_index,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                seed=attempt_seed,
            )
            attempt_results.append(attempt)
            if attempt["success"]:
                task_success = True
                successful_attempt = attempt_index
                break

        return {
            "task_id": task.task_id,
            "split": task.split,
            "theorem_name": task.theorem_name,
            "benchmark_name": args.benchmark_name,
            "success": task_success,
            "source_file": task.source_file,
            "language": task.language,
            "dataset": task.dataset,
            "pass_k": args.pass_k,
            "attempts_run": len(attempt_results),
            "successful_attempt": successful_attempt,
            "attempt_results": attempt_results,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Lean4 theorem benchmark through an external prover command")
    parser.add_argument("--benchmark-name", type=str, default="miniF2F")
    parser.add_argument("--data-path", type=str, default="")
    parser.add_argument("--manifest-jsonl", type=str, default="")
    parser.add_argument("--write-manifest", type=str, default="")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--task-limit", type=int, default=0)
    parser.add_argument("--model-ref", type=str, default="")
    parser.add_argument("--server-url", type=str, default="")
    parser.add_argument("--command-template", type=str, required=True)
    parser.add_argument("--out-json", type=str, required=True)
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--pass-k", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.pass_k <= 0:
        raise SystemExit("--pass-k must be > 0")

    tasks = _load_tasks(args)
    results: List[Dict[str, Any]] = []
    for task_index, task in enumerate(tasks, start=1):
        results.append(_run_task(task, args, task_index))

    payload = {
        "benchmark_name": args.benchmark_name,
        "split": args.split,
        "tasks": len(results),
        "successes": sum(1 for row in results if row["success"]),
        "pass_k": args.pass_k,
        "results": results,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TAAM_BENCHMARK_VERDICT: PASSED")


if __name__ == "__main__":
    main()
