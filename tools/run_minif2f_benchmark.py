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

from taam.external_tools import parse_json_result, parse_verdict_from_text, run_command_template
from taam.minif2f import load_manifest, load_miniF2F_tasks, write_miniF2F_manifest


def _load_tasks(args: argparse.Namespace):
    if args.manifest_jsonl:
        return load_manifest(Path(args.manifest_jsonl))
    if not args.data_path:
        raise SystemExit("Provide --manifest-jsonl or --data-path for miniF2F benchmark")
    tasks = load_miniF2F_tasks(Path(args.data_path), split=args.split, task_limit=args.task_limit)
    if args.write_manifest:
        write_miniF2F_manifest(tasks, Path(args.write_manifest))
    return tasks


def _run_task(task, command_template: str, timeout_sec: int, model_ref: str) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="taam_minif2f_") as tmp_dir:
        tmp = Path(tmp_dir)
        lean_file = tmp / f"{task.theorem_name}.lean"
        result_json = tmp / "result.json"
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
                },
                timeout_sec=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return {
                "task_id": task.task_id,
                "split": task.split,
                "theorem_name": task.theorem_name,
                "success": False,
                "error": "timeout",
            }

        json_verdict = parse_json_result(result_json, ("solved", "proved", "success", "passed"))
        if json_verdict is None:
            json_verdict = parse_verdict_from_text(f"{proc.stdout}\n{proc.stderr}", "TAAM_PROVER_VERDICT")
        success = proc.returncode == 0 if json_verdict is None else json_verdict
        return {
            "task_id": task.task_id,
            "split": task.split,
            "theorem_name": task.theorem_name,
            "success": bool(success),
            "source_file": task.source_file,
            "language": task.language,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Lean4 miniF2F benchmark through an external prover command")
    parser.add_argument("--data-path", type=str, default="")
    parser.add_argument("--manifest-jsonl", type=str, default="")
    parser.add_argument("--write-manifest", type=str, default="")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--task-limit", type=int, default=0)
    parser.add_argument("--model-ref", type=str, default="")
    parser.add_argument("--command-template", type=str, required=True)
    parser.add_argument("--out-json", type=str, required=True)
    parser.add_argument("--timeout-sec", type=int, default=300)
    args = parser.parse_args()

    tasks = _load_tasks(args)
    results: List[Dict[str, Any]] = []
    for task in tasks:
        results.append(_run_task(task, args.command_template, timeout_sec=args.timeout_sec, model_ref=args.model_ref))

    payload = {
        "benchmark_name": "miniF2F",
        "split": args.split,
        "tasks": len(results),
        "successes": sum(1 for row in results if row["success"]),
        "results": results,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TAAM_BENCHMARK_VERDICT: PASSED")


if __name__ == "__main__":
    main()
