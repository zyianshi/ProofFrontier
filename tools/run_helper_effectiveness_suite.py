from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taam.downstream.effectiveness_suite import summarize_effectiveness_suite
from taam.downstream.minif2f import load_benchmark_tasks, write_benchmark_manifest
from taam.downstream.repair_helper import is_algebra_like_task_name


def _build_minif2f_manifests(data_path: Path, out_dir: Path) -> tuple[Path, Path]:
    tasks = load_benchmark_tasks(data_path, dataset_name="miniF2F", split="test")
    if not tasks:
        raise SystemExit(f"No miniF2F test tasks found in {data_path}")
    algebra_like = [
        task
        for task in tasks
        if is_algebra_like_task_name(task.theorem_name) or is_algebra_like_task_name(task.task_id)
    ]
    full_manifest = out_dir / "miniF2F_full_pass32.jsonl"
    algebra_manifest = out_dir / "miniF2F_algebra_like_pass32.jsonl"
    write_benchmark_manifest(tasks, full_manifest)
    write_benchmark_manifest(algebra_like, algebra_manifest)
    return full_manifest, algebra_manifest


def _run_parallel_benchmark(
    *,
    manifest_jsonl: Path,
    model_ref: str,
    out_json: Path,
    command_template: str,
    python_executable: str,
    gpu_devices: str,
    ports: str,
    hint_source: str,
    helper_model_dir: str,
    premise_inventory_jsonl: str,
    mode: str,
    budget_schedule: str,
    vanilla_attempts: int,
    repair_attempts: int,
    timeout_sec: int,
    pass_k: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    seed: int,
    helper_device: str,
) -> None:
    command = [
        python_executable,
        "tools/run_repair_augmented_benchmark_parallel.py",
        "--benchmark-name",
        "miniF2F",
        "--manifest-jsonl",
        str(manifest_jsonl),
        "--split",
        "test",
        "--model-ref",
        model_ref,
        "--hint-source",
        hint_source,
        "--command-template",
        command_template,
        "--out-json",
        str(out_json),
        "--timeout-sec",
        str(timeout_sec),
        "--pass-k",
        str(pass_k),
        "--temperature",
        str(temperature),
        "--top-p",
        str(top_p),
        "--max-new-tokens",
        str(max_new_tokens),
        "--seed",
        str(seed),
        "--mode",
        mode,
        "--budget-schedule",
        budget_schedule,
        "--vanilla-attempts",
        str(vanilla_attempts),
        "--repair-attempts",
        str(repair_attempts),
        "--gpu-devices",
        gpu_devices,
        "--ports",
        ports,
        "--python-executable",
        python_executable,
        "--helper-device",
        helper_device,
    ]
    if helper_model_dir:
        command.extend(["--helper-model-dir", helper_model_dir])
    if premise_inventory_jsonl:
        command.extend(["--premise-inventory-jsonl", premise_inventory_jsonl])
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final hard-sample effectiveness suite")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--minif2f-data-path", type=str, required=True)
    parser.add_argument("--model-ref", type=str, default="deepseek-ai/DeepSeek-Prover-V2-7B")
    parser.add_argument("--helper-model-dir", type=str, required=True)
    parser.add_argument("--premise-inventory-jsonl", type=str, required=True)
    parser.add_argument("--hard-replay-vanilla-json", type=str, required=True)
    parser.add_argument("--hard-replay-oracle-json", type=str, required=True)
    parser.add_argument("--hard-replay-learned-json", type=str, required=True)
    parser.add_argument("--gpu-devices", type=str, default="0,1")
    parser.add_argument("--ports", type=str, default="18772,18773")
    parser.add_argument("--python-executable", type=str, default=sys.executable)
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--pass-k", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--budget-schedule", type=str, default="24/8")
    parser.add_argument("--helper-device", type=str, default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    full_manifest, algebra_manifest = _build_minif2f_manifests(Path(args.minif2f_data_path), out_dir)
    command_template = (
        'python tools/run_deepseek_prover.py --server-url "{server_url}" --model-id "{model_ref}" '
        '--lean-file "{lean_file}" --out-json "{result_json}" --timeout-sec {timeout_sec} '
        "--temperature {temperature} --top-p {top_p} --max-new-tokens {max_new_tokens} "
        '--seed {seed} --validator-project-root "/home/zhengyan/GenAI/deps/mathlib4_v420_user" --torch-dtype bfloat16'
    )

    vanilla_json = out_dir / "miniF2F_full_vanilla.json"
    learned_json = out_dir / "miniF2F_full_learned.json"
    if not vanilla_json.exists():
        _run_parallel_benchmark(
            manifest_jsonl=full_manifest,
            model_ref=args.model_ref,
            out_json=vanilla_json,
            command_template=command_template,
            python_executable=args.python_executable,
            gpu_devices=args.gpu_devices,
            ports=args.ports,
            hint_source="random",
            helper_model_dir="",
            premise_inventory_jsonl=args.premise_inventory_jsonl,
            mode="repair_helper_budget32",
            budget_schedule=args.budget_schedule,
            vanilla_attempts=args.pass_k,
            repair_attempts=0,
            timeout_sec=args.timeout_sec,
            pass_k=args.pass_k,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            helper_device=args.helper_device,
        )
    if not learned_json.exists():
        _run_parallel_benchmark(
            manifest_jsonl=full_manifest,
            model_ref=args.model_ref,
            out_json=learned_json,
            command_template=command_template,
            python_executable=args.python_executable,
            gpu_devices=args.gpu_devices,
            ports=args.ports,
            hint_source="learned",
            helper_model_dir=args.helper_model_dir,
            premise_inventory_jsonl=args.premise_inventory_jsonl,
            mode="repair_helper_budget32",
            budget_schedule=args.budget_schedule,
            vanilla_attempts=0,
            repair_attempts=args.pass_k,
            timeout_sec=args.timeout_sec,
            pass_k=args.pass_k,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            helper_device=args.helper_device,
        )

    hard_vanilla = json.loads(Path(args.hard_replay_vanilla_json).read_text(encoding="utf-8"))
    hard_oracle = json.loads(Path(args.hard_replay_oracle_json).read_text(encoding="utf-8"))
    hard_learned = json.loads(Path(args.hard_replay_learned_json).read_text(encoding="utf-8"))
    mini_vanilla = json.loads(vanilla_json.read_text(encoding="utf-8"))
    mini_learned = json.loads(learned_json.read_text(encoding="utf-8"))

    summary = summarize_effectiveness_suite(
        hard_replay_vanilla=hard_vanilla,
        hard_replay_oracle=hard_oracle,
        hard_replay_learned=hard_learned,
        minif2f_vanilla=mini_vanilla,
        minif2f_learned=mini_learned,
    )
    summary["artifacts"] = {
        "miniF2F_full_manifest": str(full_manifest),
        "miniF2F_algebra_like_manifest": str(algebra_manifest),
        "miniF2F_full_vanilla_json": str(vanilla_json),
        "miniF2F_full_learned_json": str(learned_json),
        "hard_replay_vanilla_json": args.hard_replay_vanilla_json,
        "hard_replay_oracle_json": args.hard_replay_oracle_json,
        "hard_replay_learned_json": args.hard_replay_learned_json,
    }
    out_path = out_dir / "effectiveness_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
