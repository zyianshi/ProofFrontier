from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taam.downstream.minif2f import load_benchmark_tasks, load_manifest, write_benchmark_manifest
from taam.downstream.repair_benchmark import merge_repair_benchmark_payloads, partition_repair_tasks


def _urlopen_no_proxy(req_or_url, timeout_sec: int = 30):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(req_or_url, timeout=timeout_sec)


def _parse_csv_ints(text: str) -> List[int]:
    values = [item.strip() for item in (text or "").split(",") if item.strip()]
    if not values:
        raise SystemExit("Expected a non-empty comma-separated list")
    return [int(item) for item in values]


def _load_tasks(args: argparse.Namespace):
    if args.manifest_jsonl:
        return load_manifest(Path(args.manifest_jsonl))
    if not args.data_path:
        raise SystemExit("Provide --manifest-jsonl or --data-path for repair benchmark")
    tasks = load_benchmark_tasks(
        Path(args.data_path),
        dataset_name=args.benchmark_name,
        split=args.split,
        task_limit=args.task_limit,
    )
    if args.write_manifest:
        write_benchmark_manifest(tasks, Path(args.write_manifest))
    return tasks


def _wait_for_service(url: str, timeout_sec: int) -> None:
    deadline = time.time() + timeout_sec
    last_error = ""
    while time.time() < deadline:
        try:
            with _urlopen_no_proxy(f"{url}/health", timeout_sec=10) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            if payload.get("load_error"):
                raise SystemExit(f"resident service load_error: {payload['load_error']}")
            if payload.get("loaded"):
                return
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(2)
    raise SystemExit(f"resident service did not become ready at {url}: {last_error}")


def _start_service(
    *,
    python_executable: str,
    gpu_device: int,
    port: int,
    model_ref: str,
    torch_dtype: str,
    log_path: Path,
) -> subprocess.Popen[str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    env["PYTHONUNBUFFERED"] = "1"
    log_handle = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(
        [
            python_executable,
            "tools/run_deepseek_prover.py",
            "--serve",
            "--model-id",
            model_ref,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--cuda-device",
            "0",
            "--torch-dtype",
            torch_dtype,
        ],
        cwd=ROOT,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _build_shard_command(
    *,
    python_executable: str,
    shard_manifest: Path,
    shard_out_json: Path,
    shard_log: Path,
    args: argparse.Namespace,
    server_url: str,
) -> tuple[List[str], Path]:
    command = [
        python_executable,
        "tools/run_repair_augmented_benchmark.py",
        "--benchmark-name",
        args.benchmark_name,
        "--manifest-jsonl",
        str(shard_manifest),
        "--split",
        args.split,
        "--model-ref",
        args.model_ref,
        "--server-url",
        server_url,
        "--command-template",
        args.command_template,
        "--out-json",
        str(shard_out_json),
        "--timeout-sec",
        str(args.timeout_sec),
        "--pass-k",
        str(args.pass_k),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--seed",
        str(args.seed),
        "--mode",
        args.mode,
        "--budget-schedule",
        args.budget_schedule,
        "--vanilla-attempts",
        str(args.vanilla_attempts),
        "--repair-attempts",
        str(args.repair_attempts),
        "--hint-source",
        args.hint_source,
    ]
    if args.helper_model_dir:
        command.extend(["--helper-model-dir", args.helper_model_dir])
    if args.premise_inventory_jsonl:
        command.extend(["--premise-inventory-jsonl", args.premise_inventory_jsonl])
    if args.bm25_candidate_count:
        command.extend(["--bm25-candidate-count", str(args.bm25_candidate_count)])
    if args.rerank_top_n:
        command.extend(["--rerank-top-n", str(args.rerank_top_n)])
    if args.hint_top_k:
        command.extend(["--hint-top-k", str(args.hint_top_k)])
    if args.max_length:
        command.extend(["--max-length", str(args.max_length)])
    if args.helper_device:
        command.extend(["--helper-device", args.helper_device])
    return command, shard_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repair benchmark in parallel across multiple GPUs")
    parser.add_argument("--benchmark-name", type=str, required=True)
    parser.add_argument("--data-path", type=str, default="")
    parser.add_argument("--manifest-jsonl", type=str, default="")
    parser.add_argument("--write-manifest", type=str, default="")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--task-limit", type=int, default=0)
    parser.add_argument("--model-ref", type=str, required=True)
    parser.add_argument("--helper-model-dir", type=str, default="")
    parser.add_argument("--premise-inventory-jsonl", type=str, default="")
    parser.add_argument("--hint-source", type=str, default="learned")
    parser.add_argument("--command-template", type=str, required=True)
    parser.add_argument("--out-json", type=str, required=True)
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--pass-k", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mode", type=str, default="repair_helper_budget32")
    parser.add_argument("--budget-schedule", type=str, default="24/8")
    parser.add_argument("--vanilla-attempts", type=int, default=0)
    parser.add_argument("--repair-attempts", type=int, default=0)
    parser.add_argument("--bm25-candidate-count", type=int, default=0)
    parser.add_argument("--rerank-top-n", type=int, default=0)
    parser.add_argument("--hint-top-k", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=0)
    parser.add_argument("--helper-device", type=str, default="cpu")
    parser.add_argument("--gpu-devices", type=str, required=True, help="Comma-separated physical GPU ids, e.g. 0,1")
    parser.add_argument("--ports", type=str, required=True, help="Comma-separated ports, one per GPU")
    parser.add_argument("--python-executable", type=str, default=sys.executable)
    parser.add_argument("--resident-torch-dtype", type=str, default="bfloat16")
    parser.add_argument("--service-startup-timeout-sec", type=int, default=900)
    args = parser.parse_args()

    gpu_devices = _parse_csv_ints(args.gpu_devices)
    ports = _parse_csv_ints(args.ports)
    if len(gpu_devices) != len(ports):
        raise SystemExit("--gpu-devices and --ports must have the same length")

    tasks = _load_tasks(args)
    if not tasks:
        raise SystemExit("No benchmark tasks available for parallel run")
    ordered_task_ids = [task.task_id for task in tasks]
    shards = [shard for shard in partition_repair_tasks(tasks, len(gpu_devices)) if shard]
    if not shards:
        raise SystemExit("Parallel task partitioning produced no shards")

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    services: List[subprocess.Popen[str]] = []
    runners: List[subprocess.Popen[str]] = []
    log_handles = []
    try:
        shard_specs = []
        for shard_index, shard_tasks in enumerate(shards):
            gpu_device = gpu_devices[shard_index]
            port = ports[shard_index]
            service_url = f"http://127.0.0.1:{port}"
            shard_manifest = out_path.parent / f"{out_path.stem}.shard{shard_index:02d}.manifest.jsonl"
            shard_out_json = out_path.parent / f"{out_path.stem}.shard{shard_index:02d}.json"
            shard_log = out_path.parent / f"{out_path.stem}.shard{shard_index:02d}.log"
            service_log = out_path.parent / f"{out_path.stem}.service{shard_index:02d}.log"
            write_benchmark_manifest(shard_tasks, shard_manifest)
            service_proc = _start_service(
                python_executable=args.python_executable,
                gpu_device=gpu_device,
                port=port,
                model_ref=args.model_ref,
                torch_dtype=args.resident_torch_dtype,
                log_path=service_log,
            )
            services.append(service_proc)
            shard_specs.append((service_url, shard_manifest, shard_out_json, shard_log))

        for service_url, _, _, _ in shard_specs:
            _wait_for_service(service_url, timeout_sec=args.service_startup_timeout_sec)

        for service_url, shard_manifest, shard_out_json, shard_log in shard_specs:
            command, _ = _build_shard_command(
                python_executable=args.python_executable,
                shard_manifest=shard_manifest,
                shard_out_json=shard_out_json,
                shard_log=shard_log,
                args=args,
                server_url=service_url,
            )
            handle = shard_log.open("w", encoding="utf-8")
            log_handles.append(handle)
            runners.append(
                subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            )

        exit_codes = [proc.wait() for proc in runners]
        if any(code != 0 for code in exit_codes):
            raise SystemExit(f"At least one shard benchmark failed: {exit_codes}")

        payloads = [json.loads(shard_out_json.read_text(encoding="utf-8")) for _, _, shard_out_json, _ in shard_specs]
        merged = merge_repair_benchmark_payloads(payloads, ordered_task_ids=ordered_task_ids)
        out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        print("TAAM_BENCHMARK_VERDICT: PASSED")
    finally:
        for proc in runners:
            if proc.poll() is None:
                proc.kill()
        for proc in services:
            if proc.poll() is None:
                proc.kill()
        for handle in log_handles:
            handle.close()


if __name__ == "__main__":
    main()
