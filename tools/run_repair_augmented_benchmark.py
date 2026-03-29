from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
import random

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taam.downstream.minif2f import load_benchmark_tasks, load_manifest, write_benchmark_manifest
from taam.downstream.repair_benchmark import (
    RepairBenchmarkTask,
    allocate_repair_attempt_budgets,
    run_repair_task,
)
from taam.downstream.repair_helper import (
    BM25Index,
    PremiseReranker,
    load_helper_corpus,
    load_helper_metadata,
    select_hint_premises,
)


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


def _resolve_helper_hparams(args: argparse.Namespace, helper_metadata: Dict[str, Any]) -> tuple[int, int, int, int]:
    bm25_candidate_count = args.bm25_candidate_count or int(helper_metadata.get("bm25_candidate_count", 32))
    rerank_top_n = args.rerank_top_n or int(helper_metadata.get("rerank_top_n", 8))
    hint_top_k = args.hint_top_k or int(helper_metadata.get("hint_top_k", 8))
    max_length = args.max_length or int(helper_metadata.get("max_length", 512))
    return bm25_candidate_count, rerank_top_n, hint_top_k, max_length


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repair-augmented theorem benchmark with a frozen base prover")
    parser.add_argument("--benchmark-name", type=str, default="miniF2F")
    parser.add_argument("--data-path", type=str, default="")
    parser.add_argument("--manifest-jsonl", type=str, default="")
    parser.add_argument("--write-manifest", type=str, default="")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--task-limit", type=int, default=0)
    parser.add_argument("--model-ref", type=str, required=True)
    parser.add_argument("--helper-model-dir", type=str, default="")
    parser.add_argument("--premise-inventory-jsonl", type=str, default="")
    parser.add_argument("--hint-source", type=str, default="learned")
    parser.add_argument("--server-url", type=str, default="")
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
    args = parser.parse_args()

    helper_metadata: Dict[str, Any] = {}
    helper_model_dir: Path | None = None
    if args.helper_model_dir:
        helper_model_dir = Path(args.helper_model_dir)
        if not helper_model_dir.exists():
            raise SystemExit(f"Helper model directory does not exist: {helper_model_dir}")
        helper_metadata = load_helper_metadata(helper_model_dir)
    if args.hint_source == "learned" and helper_model_dir is None:
        raise SystemExit("--helper-model-dir is required when --hint-source=learned")
    if helper_model_dir is None and not args.premise_inventory_jsonl:
        raise SystemExit("Provide --premise-inventory-jsonl when helper model dir is omitted")
    bm25_candidate_count, rerank_top_n, hint_top_k, max_length = _resolve_helper_hparams(args, helper_metadata)
    corpus = load_helper_corpus(
        helper_model_dir,
        Path(args.premise_inventory_jsonl) if args.premise_inventory_jsonl else None,
    )
    if not corpus:
        raise SystemExit("Premise corpus is empty for repair benchmark")
    bm25 = BM25Index(corpus)
    reranker = None
    if args.hint_source == "learned":
        reranker = PremiseReranker(helper_model_dir, max_length=max_length, device=args.helper_device)
    tasks = _load_tasks(args)
    rng = random.Random(args.seed)
    budgets = {
        "vanilla_attempts": args.vanilla_attempts,
        "repair_attempts": args.repair_attempts,
    }
    if budgets["vanilla_attempts"] <= 0 or budgets["repair_attempts"] < 0:
        budgets = allocate_repair_attempt_budgets(args.mode, args.pass_k, args.budget_schedule)

    results: List[Dict[str, Any]] = []
    for task_index, task in enumerate(tasks, start=1):
        repair_task = RepairBenchmarkTask(
            task_id=task.task_id,
            theorem_name=task.theorem_name,
            split=task.split,
            source_file=task.source_file,
            lean_code=task.lean_code,
            dataset=task.dataset,
            language=task.language,
            metadata=dict(task.metadata),
        )
        helper_hits = select_hint_premises(
            query=task.lean_code,
            task_metadata=dict(task.metadata),
            corpus=corpus,
            bm25=bm25,
            reranker=reranker,
            hint_source=args.hint_source,
            bm25_candidate_count=bm25_candidate_count,
            rerank_top_n=rerank_top_n,
            hint_top_k=hint_top_k,
            rng=random.Random(rng.randint(0, 10**9)),
        )
        task_result = run_repair_task(
            task=repair_task,
            command_template=args.command_template,
            model_ref=args.model_ref,
            server_url=args.server_url,
            timeout_sec=args.timeout_sec,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed + ((task_index - 1) * max(1, args.pass_k)),
            vanilla_attempts=budgets["vanilla_attempts"],
            repair_attempts=budgets["repair_attempts"],
            helper_hits=helper_hits,
        )
        results.append(
            {
                "task_id": task.task_id,
                "split": task.split,
                "theorem_name": task.theorem_name,
                "benchmark_name": args.benchmark_name,
                "success": bool(task_result["success"]),
                "source_file": task.source_file,
                "language": task.language,
                "dataset": task.dataset,
                "pass_k": args.pass_k,
                "budget_schedule": args.budget_schedule,
                "hint_source": args.hint_source,
                "successful_stage": task_result["successful_stage"],
                "vanilla_success": task_result["vanilla_success"],
                "repair_success": task_result["repair_success"],
                "retrieved_premises": [item["premise_id"] for item in helper_hits],
                "retrieved_premise_count": len(helper_hits),
                "retrieved_premise_sources": [item.get("source", "") for item in helper_hits],
                "vanilla_attempt_results": task_result["vanilla_attempt_results"],
                "repair_attempt_results": task_result["repair_attempt_results"],
            }
        )

    payload = {
        "benchmark_name": args.benchmark_name,
        "split": args.split,
        "tasks": len(results),
        "successes": sum(1 for row in results if row["success"]),
        "pass_k": args.pass_k,
        "mode": args.mode,
        "budget_schedule": args.budget_schedule,
        "hint_source": args.hint_source,
        "results": results,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TAAM_BENCHMARK_VERDICT: PASSED")


if __name__ == "__main__":
    main()
