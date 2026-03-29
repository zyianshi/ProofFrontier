from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..external_tools import materialize_lean_completion, run_command_template


def collect_hard_sample_paths(samples_root: Path, sample_glob: str) -> List[Path]:
    return sorted(samples_root.glob(sample_glob))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_hard_samples(samples_root: Path, sample_glob: str) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for path in collect_hard_sample_paths(samples_root, sample_glob):
        sample = load_json(path)
        sample["_source_path"] = str(path)
        samples.append(sample)
    return samples


def load_traced_inventory_rows(inventory_jsonl: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with inventory_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_source_path"] = str(inventory_jsonl)
            rows.append(row)
    return rows


def _extract_prompt_from_lean_problem(lean_problem: str) -> str:
    if "sorry" not in lean_problem:
        return lean_problem
    return lean_problem.replace("sorry", "", 1).rstrip() + "\n"


def _normalize_completion(completion: str) -> str:
    return completion.strip() + ("\n" if completion.strip() else "")


def _split_namespace(theorem_id: str) -> tuple[str, str]:
    parts = str(theorem_id).strip().split(".")
    if len(parts) <= 1:
        return "", theorem_id
    return ".".join(parts[:-1]), parts[-1]


def _safe_ident(name: str) -> str:
    clean = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in str(name))
    return clean or "replay_theorem"


def _build_source_proof_problem(theorem_id: str, theorem_context: Sequence[str], target_statement: str) -> str:
    lines: List[str] = ["import Mathlib", ""]
    namespace_name, theorem_tail = _split_namespace(theorem_id)
    theorem_name = _safe_ident(f"replay_{theorem_tail}")
    context_chunks = [str(chunk).strip() for chunk in theorem_context if str(chunk).strip()]
    if namespace_name:
        lines.append(f"namespace {namespace_name}")
        lines.append("")
    signature = " ".join(context_chunks)
    header = f"theorem {theorem_name}"
    if signature:
        header += f" {signature}"
    header += f" : {target_statement} := by"
    lines.append(header)
    lines.append("  sorry")
    if namespace_name:
        lines.append("")
        lines.append(f"end {namespace_name}")
    return "\n".join(lines)


def sample_to_record(sample: Dict[str, Any], dataset_format: str) -> Optional[Dict[str, Any]]:
    lean_problem = str(sample.get("lean_problem", ""))
    if not lean_problem.strip():
        return None

    record = {
        "sample_id": f'{sample.get("theorem_id", "unknown")}::{sample.get("target_id", "T")}::{sample.get("failed_on", "")}',
        "theorem_id": sample.get("theorem_id", ""),
        "target_id": sample.get("target_id", ""),
        "failed_on": sample.get("failed_on", ""),
        "hidden_nodes": list(sample.get("hidden_nodes", [])),
        "visible_nodes": list(sample.get("visible_nodes", [])),
        "masking_strategy": sample.get("masking_strategy", ""),
        "lean_problem": lean_problem,
        "masked_lean_problem": lean_problem,
        "full_lean_problem": str(sample.get("full_lean_problem", "")).strip(),
        "proof_completion": str(sample.get("proof_completion", "")).strip(),
        "well_posed": sample.get("well_posed", None),
        "proof_source": sample.get("proof_source", ""),
        "source_path": sample.get("_source_path", ""),
        "theorem_domain": sample.get("theorem_domain", ""),
        "source_file_path": sample.get("source_file_path", ""),
    }

    fmt = dataset_format.lower().strip()
    if fmt == "rl":
        record.update(
            {
                "task_type": "lean_proof_search",
                "prompt": lean_problem,
                "supervision_mode": "policy_optimization_on_masked_problem",
                "reward_spec": {
                    "type": "lean_compile_and_prove",
                    "success_field": "passed",
                    "notes": "Reward should come from Lean validation or benchmark prover success.",
                },
            }
        )
        return record
    if fmt == "sft":
        proof_completion = str(sample.get("proof_completion", "")).strip()
        if not proof_completion:
            return None
        record.update(
            {
                "task_type": "lean_proof_completion",
                "prompt": _extract_prompt_from_lean_problem(lean_problem),
                "completion": _normalize_completion(proof_completion),
                "ground_truth_proof": _normalize_completion(proof_completion),
                "supervision_mode": "full_proof_from_original_complete_proof_chain",
            }
        )
        return record
    if fmt == "helper":
        positives = []
        for item in [str(sample.get("failed_on", "")).strip(), *[str(x) for x in sample.get("hidden_nodes", [])]]:
            if item and item not in positives:
                positives.append(item)
        negatives = []
        for item in [str(x) for x in sample.get("visible_nodes", [])]:
            if item and item not in positives and item not in negatives:
                negatives.append(item)
        record.update(
            {
                "task_type": "premise_repair",
                "query": lean_problem,
                "positive_premises": sorted(positives),
                "negative_premises": negatives,
            }
        )
        return record
    raise ValueError(f"Unsupported dataset format: {dataset_format}")


def inventory_row_to_record(row: Dict[str, Any], dataset_format: str) -> Optional[Dict[str, Any]]:
    theorem_id = str(row.get("theorem_id", "")).strip()
    target_statement = str(row.get("target_statement", "")).strip()
    proof_completion = str(row.get("tactic_proof", row.get("proof_completion", ""))).strip()
    theorem_context = list(row.get("theorem_context", []))
    if not theorem_id or not target_statement:
        return None

    lean_problem = _build_source_proof_problem(theorem_id, theorem_context, target_statement)
    record = {
        "sample_id": theorem_id,
        "theorem_id": theorem_id,
        "target_id": theorem_id,
        "failed_on": "",
        "hidden_nodes": [],
        "visible_nodes": [],
        "masking_strategy": "source_proof",
        "lean_problem": lean_problem,
        "masked_lean_problem": lean_problem,
        "full_lean_problem": materialize_lean_completion(lean_problem, proof_completion) if proof_completion else lean_problem,
        "proof_completion": proof_completion,
        "well_posed": True,
        "proof_source": str(row.get("source", "traced_inventory")),
        "source_path": str(row.get("_source_path", "")),
        "source_file_path": str(row.get("file_path", "")),
    }
    fmt = dataset_format.lower().strip()
    if fmt == "sft":
        if not proof_completion:
            return None
        record.update(
            {
                "task_type": "lean_proof_completion",
                "prompt": _extract_prompt_from_lean_problem(lean_problem),
                "completion": _normalize_completion(proof_completion),
                "ground_truth_proof": _normalize_completion(proof_completion),
                "supervision_mode": "full_proof_from_traced_inventory",
            }
        )
        return record
    if fmt == "helper":
        record.update(
            {
                "task_type": "premise_document",
                "premise_id": theorem_id,
                "target_statement": target_statement,
                "text": f"{theorem_id} : {target_statement}",
            }
        )
        return record
    if fmt == "rl":
        record.update(
            {
                "task_type": "lean_proof_search",
                "prompt": lean_problem,
                "supervision_mode": "policy_optimization_on_source_problem",
            }
        )
        return record
    raise ValueError(f"Unsupported dataset format: {dataset_format}")


def build_dataset_records(
    samples: Sequence[Dict[str, Any]],
    dataset_format: str,
    only_well_posed: bool = True,
    require_proof_completion: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    stats = {
        "total_samples": len(samples),
        "kept_samples": 0,
        "skipped_not_well_posed": 0,
        "skipped_missing_lean_problem": 0,
        "skipped_missing_proof_completion": 0,
        "skipped_missing_positive_premises": 0,
    }
    for sample in samples:
        if only_well_posed and sample.get("well_posed", None) is not True:
            stats["skipped_not_well_posed"] += 1
            continue
        if not str(sample.get("lean_problem", "")).strip():
            stats["skipped_missing_lean_problem"] += 1
            continue
        if require_proof_completion and not str(sample.get("proof_completion", "")).strip():
            stats["skipped_missing_proof_completion"] += 1
            continue
        record = sample_to_record(sample, dataset_format)
        if record is None:
            stats["skipped_missing_proof_completion"] += 1
            continue
        if dataset_format.lower().strip() == "helper" and not record.get("positive_premises"):
            stats["skipped_missing_positive_premises"] += 1
            continue
        records.append(record)
    stats["kept_samples"] = len(records)
    return records, stats


def build_inventory_dataset_records(
    rows: Sequence[Dict[str, Any]],
    dataset_format: str,
    require_proof_completion: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    stats = {
        "total_samples": len(rows),
        "kept_samples": 0,
        "skipped_missing_target_statement": 0,
        "skipped_missing_proof_completion": 0,
        "skipped_missing_theorem_id": 0,
    }
    for row in rows:
        theorem_id = str(row.get("theorem_id", "")).strip()
        target_statement = str(row.get("target_statement", "")).strip()
        proof_completion = str(row.get("tactic_proof", row.get("proof_completion", ""))).strip()
        if not theorem_id:
            stats["skipped_missing_theorem_id"] += 1
            continue
        if not target_statement:
            stats["skipped_missing_target_statement"] += 1
            continue
        if require_proof_completion and not proof_completion:
            stats["skipped_missing_proof_completion"] += 1
            continue
        record = inventory_row_to_record(row, dataset_format)
        if record is None:
            stats["skipped_missing_proof_completion"] += 1
            continue
        records.append(record)
    stats["kept_samples"] = len(records)
    return records, stats


def split_records(
    records: Sequence[Dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[Dict[str, Any]]]:
    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("train_ratio + val_ratio + test_ratio must be > 0")
    normalized = [train_ratio / total_ratio, val_ratio / total_ratio, test_ratio / total_ratio]
    data = list(records)
    random.Random(seed).shuffle(data)
    n = len(data)
    n_train = int(n * normalized[0])
    n_val = int(n * normalized[1])
    return {
        "train": data[:n_train],
        "val": data[n_train : n_train + n_val],
        "test": data[n_train + n_val :],
    }


def write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def export_dataset_bundle(
    source_type: str,
    samples_root: Path,
    sample_glob: str,
    inventory_jsonl: Path,
    out_dir: Path,
    dataset_format: str,
    only_well_posed: bool,
    require_proof_completion: bool,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, Any]:
    if source_type == "hard_samples":
        samples = load_hard_samples(samples_root, sample_glob)
        records, stats = build_dataset_records(
            samples,
            dataset_format=dataset_format,
            only_well_posed=only_well_posed,
            require_proof_completion=require_proof_completion,
        )
    elif source_type == "traced_inventory":
        rows = load_traced_inventory_rows(inventory_jsonl)
        records, stats = build_inventory_dataset_records(
            rows,
            dataset_format=dataset_format,
            require_proof_completion=require_proof_completion,
        )
    else:
        raise ValueError(f"Unsupported downstream dataset source_type: {source_type}")

    splits = split_records(records, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_paths: Dict[str, str] = {}
    for split_name, split_rows in splits.items():
        split_path = out_dir / f"{split_name}.jsonl"
        write_jsonl(split_rows, split_path)
        split_paths[split_name] = str(split_path)
    manifest = {
        "source_type": source_type,
        "dataset_format": dataset_format,
        "samples_root": str(samples_root),
        "sample_glob": sample_glob,
        "inventory_jsonl": str(inventory_jsonl),
        "seed": seed,
        "split_paths": split_paths,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "stats": stats,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _load_rows_from_path(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        data = load_json(path)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if isinstance(data.get("records"), list):
                return data["records"]
            if isinstance(data.get("results"), list):
                return data["results"]
        raise ValueError(f"Unsupported JSON benchmark result schema: {path}")
    raise ValueError(f"Unsupported benchmark result file type: {path}")


def load_benchmark_rows(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        rows.extend(_load_rows_from_path(path))
    return rows


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "passed", "proved", "success"}


def _row_id(row: Dict[str, Any]) -> str:
    for key in ("task_id", "theorem_id", "name", "id"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    raise ValueError("Benchmark row is missing task identifier field")


def _row_split(row: Dict[str, Any]) -> str:
    return str(row.get("split", "all") or "all")


def _row_success(row: Dict[str, Any]) -> bool:
    for key in ("success", "passed", "proved", "solved"):
        if key in row:
            return _as_bool(row[key])
    return False


def summarize_benchmark_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    successes = sum(1 for row in rows if _row_success(row))
    by_split: Dict[str, List[bool]] = defaultdict(list)
    for row in rows:
        by_split[_row_split(row)].append(_row_success(row))
    return {
        "tasks": total,
        "successes": successes,
        "pass_rate": (successes / total) if total else 0.0,
        "by_split": {
            split: {
                "tasks": len(values),
                "successes": sum(1 for value in values if value),
                "pass_rate": (sum(1 for value in values if value) / len(values)) if values else 0.0,
            }
            for split, values in sorted(by_split.items())
        },
    }


def compare_benchmark_runs(before_rows: Sequence[Dict[str, Any]], after_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    before_map = {_row_id(row): row for row in before_rows}
    after_map = {_row_id(row): row for row in after_rows}
    shared_ids = sorted(set(before_map) & set(after_map))
    if not shared_ids:
        raise ValueError("No overlapping task ids between before/after benchmark results")

    improvements = 0
    regressions = 0
    unchanged = 0
    split_buckets: Dict[str, List[Tuple[bool, bool]]] = defaultdict(list)
    for task_id in shared_ids:
        before_ok = _row_success(before_map[task_id])
        after_ok = _row_success(after_map[task_id])
        split_name = _row_split(after_map[task_id] if _row_split(after_map[task_id]) != "all" else before_map[task_id])
        split_buckets[split_name].append((before_ok, after_ok))
        if (not before_ok) and after_ok:
            improvements += 1
        elif before_ok and (not after_ok):
            regressions += 1
        else:
            unchanged += 1

    before_summary = summarize_benchmark_rows([before_map[task_id] for task_id in shared_ids])
    after_summary = summarize_benchmark_rows([after_map[task_id] for task_id in shared_ids])
    error_before = 1.0 - before_summary["pass_rate"]
    error_after = 1.0 - after_summary["pass_rate"]
    return {
        "matched_tasks": len(shared_ids),
        "before": before_summary,
        "after": after_summary,
        "absolute_gain": after_summary["pass_rate"] - before_summary["pass_rate"],
        "relative_gain": (
            (after_summary["pass_rate"] - before_summary["pass_rate"]) / before_summary["pass_rate"]
            if before_summary["pass_rate"] > 0
            else None
        ),
        "error_reduction": ((error_before - error_after) / error_before) if error_before > 0 else None,
        "improved_tasks": improvements,
        "regressed_tasks": regressions,
        "unchanged_tasks": unchanged,
        "win_rate": improvements / len(shared_ids),
        "regression_rate": regressions / len(shared_ids),
    }


def run_training_job(command_template: str, placeholders: Dict[str, str], timeout_sec: int) -> Dict[str, Any]:
    proc = run_command_template(command_template, placeholders, timeout_sec=timeout_sec)
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "success": proc.returncode == 0,
    }


def run_benchmark_job(command_template: str, placeholders: Dict[str, str], timeout_sec: int) -> Dict[str, Any]:
    proc = run_command_template(command_template, placeholders, timeout_sec=timeout_sec)
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "success": proc.returncode == 0,
    }
