from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class ExperimentConfig:
    name: str = "default"
    graph_json: str = ""
    lean_trace_json: str = ""
    out_dir: str = "results/midstream_taam_generation/runs"
    capability_threshold: float = 0.0
    seed: int = 7
    solver_type: str = "deepseek_prover"
    solver_config: Dict[str, Any] = field(default_factory=dict)
    validation_config: Dict[str, Any] = field(default_factory=dict)
    masking_strategy: str = "greedy_topology"
    masking_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SweepConfig:
    name: str = "sweep"
    graph_json: str = ""
    graph_dir: str = ""
    graph_manifest_jsonl: str = ""
    graph_glob: str = "**/*.graph.json"
    graph_limit: int = 0
    lean_trace_json: str = ""
    out_root: str = "results/midstream_taam_generation/sweeps"
    capability_thresholds: List[float] = None
    seeds: List[int] = None
    solver_type: str = "deepseek_prover"
    solver_config: Dict[str, Any] = field(default_factory=dict)
    validation_config: Dict[str, Any] = field(default_factory=dict)
    masking_strategy: str = "greedy_topology"
    masking_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.capability_thresholds is None:
            self.capability_thresholds = [0.0]
        if self.seeds is None:
            self.seeds = [7, 13, 42]


@dataclass
class DownstreamDatasetConfig:
    source_type: str = "hard_samples"
    samples_root: str = "results/midstream_taam_generation/runs"
    sample_glob: str = "**/*_hard_sample.json"
    inventory_jsonl: str = ""
    format: str = "rl"
    only_well_posed: bool = True
    require_proof_completion: bool = False
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 7


@dataclass
class PremiseHelperConfig:
    inventory_jsonl: str = ""
    dataset: DownstreamDatasetConfig = field(default_factory=lambda: DownstreamDatasetConfig(format="helper"))
    model_name: str = "microsoft/codebert-base"
    bm25_candidate_count: int = 32
    rerank_top_n: int = 8
    hint_top_k: int = 8
    budget_schedule_candidates: List[str] = field(default_factory=lambda: ["24/8", "16/16", "8/24"])
    train_num_epochs: int = 1
    train_batch_size: int = 8
    eval_batch_size: int = 8
    learning_rate: float = 2e-5
    max_length: int = 512
    seed: int = 7


@dataclass
class DownstreamArmConfig:
    name: str = "arm"
    enabled: bool = True
    model_family: str = "prover"
    mode: str = "vanilla"
    base_model: str = ""
    benchmark_model_ref: str = ""
    training_enabled: bool = False
    training_command_template: str = ""
    training_timeout_sec: int = 7200
    output_model_dir: str = ""
    compare_to: str = ""
    benchmark_command_template: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    reuse_training_from: str = ""


@dataclass
class DownstreamBenchmarkSuiteConfig:
    name: str = "miniF2F_test"
    dataset_name: str = "miniF2F"
    split: str = "test"
    data_path: str = ""
    manifest_path: str = ""
    task_limit: int = 0
    command_template: str = ""
    timeout_sec: int = 7200
    pass_k: int = 32
    temperature: float = 1.0
    top_p: float = 0.95
    max_new_tokens: int = 2048
    seed: int = 7


@dataclass
class DownstreamBenchmarkRuntimeConfig:
    host: str = "127.0.0.1"
    port: int = 18765
    gpu_device: int = 0
    startup_timeout_sec: int = 300
    healthcheck_sec: int = 2


@dataclass
class DownstreamConfig:
    name: str = "downstream"
    out_dir: str = "results/downstream_prover_eval"
    dataset: DownstreamDatasetConfig = field(default_factory=DownstreamDatasetConfig)
    premise_helper: PremiseHelperConfig = field(default_factory=PremiseHelperConfig)
    arms: List[DownstreamArmConfig] = field(default_factory=list)
    benchmarks: List[DownstreamBenchmarkSuiteConfig] = field(default_factory=list)
    benchmark_runtime: DownstreamBenchmarkRuntimeConfig = field(default_factory=DownstreamBenchmarkRuntimeConfig)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_dataset_config(payload: Dict[str, Any], defaults: Dict[str, Any] | None = None) -> DownstreamDatasetConfig:
    defaults = defaults or {}
    return DownstreamDatasetConfig(
        source_type=payload.get("source_type", defaults.get("source_type", "hard_samples")),
        samples_root=payload.get("samples_root", defaults.get("samples_root", "results/midstream_taam_generation/runs")),
        sample_glob=payload.get("sample_glob", defaults.get("sample_glob", "**/*_hard_sample.json")),
        inventory_jsonl=payload.get("inventory_jsonl", defaults.get("inventory_jsonl", "")),
        format=payload.get("format", defaults.get("format", "rl")),
        only_well_posed=bool(payload.get("only_well_posed", defaults.get("only_well_posed", True))),
        require_proof_completion=bool(
            payload.get("require_proof_completion", defaults.get("require_proof_completion", False))
        ),
        train_ratio=float(payload.get("train_ratio", defaults.get("train_ratio", 0.8))),
        val_ratio=float(payload.get("val_ratio", defaults.get("val_ratio", 0.1))),
        test_ratio=float(payload.get("test_ratio", defaults.get("test_ratio", 0.1))),
        seed=int(payload.get("seed", defaults.get("seed", 7))),
    )


def load_experiment_config(path: Path) -> ExperimentConfig:
    data = _read_json(path)
    return ExperimentConfig(
        name=data.get("name", "default"),
        graph_json=data.get("graph_json", ""),
        lean_trace_json=data.get("lean_trace_json", ""),
        out_dir=data.get("out_dir", "results/midstream_taam_generation/runs"),
        capability_threshold=float(data.get("capability_threshold", 0.0)),
        seed=int(data.get("seed", 7)),
        solver_type=data.get("solver_type", "deepseek_prover"),
        solver_config=dict(data.get("solver_config", {})),
        validation_config=dict(data.get("validation_config", {})),
        masking_strategy=data.get("masking_strategy", "greedy_topology"),
        masking_config=dict(data.get("masking_config", {})),
    )


def load_sweep_config(path: Path) -> SweepConfig:
    data = _read_json(path)
    return SweepConfig(
        name=data.get("name", "sweep"),
        graph_json=data.get("graph_json", ""),
        graph_dir=data.get("graph_dir", ""),
        graph_manifest_jsonl=data.get("graph_manifest_jsonl", ""),
        graph_glob=data.get("graph_glob", "**/*.graph.json"),
        graph_limit=int(data.get("graph_limit", 0)),
        lean_trace_json=data.get("lean_trace_json", ""),
        out_root=data.get("out_root", "results/midstream_taam_generation/sweeps"),
        capability_thresholds=[float(x) for x in data.get("capability_thresholds", [0.0])],
        seeds=[int(x) for x in data.get("seeds", [7, 13, 42])],
        solver_type=data.get("solver_type", "deepseek_prover"),
        solver_config=dict(data.get("solver_config", {})),
        validation_config=dict(data.get("validation_config", {})),
        masking_strategy=data.get("masking_strategy", "greedy_topology"),
        masking_config=dict(data.get("masking_config", {})),
    )


def load_downstream_config(path: Path) -> DownstreamConfig:
    data = _read_json(path)
    dataset_payload = dict(data.get("dataset", {}))
    premise_helper_payload = dict(data.get("premise_helper", {}))
    arms = data.get("arms", [])
    benchmarks = data.get("benchmarks", [])
    benchmark_runtime = data.get("benchmark_runtime", {})

    dataset = _load_dataset_config(dataset_payload)
    helper_dataset = _load_dataset_config(
        dict(premise_helper_payload.get("dataset", {})),
        defaults={
            "source_type": "hard_samples",
            "samples_root": dataset.samples_root,
            "sample_glob": dataset.sample_glob,
            "inventory_jsonl": "",
            "format": "helper",
            "only_well_posed": dataset.only_well_posed,
            "require_proof_completion": False,
            "train_ratio": dataset.train_ratio,
            "val_ratio": dataset.val_ratio,
            "test_ratio": dataset.test_ratio,
            "seed": dataset.seed,
        },
    )

    return DownstreamConfig(
        name=data.get("name", "downstream"),
        out_dir=data.get("out_dir", "results/downstream_prover_eval"),
        dataset=dataset,
        premise_helper=PremiseHelperConfig(
            inventory_jsonl=premise_helper_payload.get("inventory_jsonl", ""),
            dataset=helper_dataset,
            model_name=premise_helper_payload.get("model_name", "microsoft/codebert-base"),
            bm25_candidate_count=int(premise_helper_payload.get("bm25_candidate_count", 32)),
            rerank_top_n=int(premise_helper_payload.get("rerank_top_n", 8)),
            hint_top_k=int(premise_helper_payload.get("hint_top_k", 8)),
            budget_schedule_candidates=[
                str(item) for item in premise_helper_payload.get("budget_schedule_candidates", ["24/8", "16/16", "8/24"])
            ],
            train_num_epochs=int(premise_helper_payload.get("train_num_epochs", 1)),
            train_batch_size=int(premise_helper_payload.get("train_batch_size", 8)),
            eval_batch_size=int(premise_helper_payload.get("eval_batch_size", 8)),
            learning_rate=float(premise_helper_payload.get("learning_rate", 2e-5)),
            max_length=int(premise_helper_payload.get("max_length", 512)),
            seed=int(premise_helper_payload.get("seed", helper_dataset.seed)),
        ),
        benchmark_runtime=DownstreamBenchmarkRuntimeConfig(
            host=benchmark_runtime.get("host", "127.0.0.1"),
            port=int(benchmark_runtime.get("port", 18765)),
            gpu_device=int(benchmark_runtime.get("gpu_device", 0)),
            startup_timeout_sec=int(benchmark_runtime.get("startup_timeout_sec", 300)),
            healthcheck_sec=int(benchmark_runtime.get("healthcheck_sec", 2)),
        ),
        arms=[
            DownstreamArmConfig(
                name=arm.get("name", f"arm_{idx}"),
                enabled=bool(arm.get("enabled", True)),
                model_family=arm.get("model_family", "prover"),
                mode=arm.get("mode", "vanilla"),
                base_model=arm.get("base_model", ""),
                benchmark_model_ref=arm.get("benchmark_model_ref", ""),
                training_enabled=bool(arm.get("training_enabled", False)),
                training_command_template=arm.get("training_command_template", ""),
                training_timeout_sec=int(arm.get("training_timeout_sec", 7200)),
                output_model_dir=arm.get("output_model_dir", ""),
                compare_to=arm.get("compare_to", ""),
                benchmark_command_template=arm.get("benchmark_command_template", ""),
                metadata=dict(arm.get("metadata", {})),
                reuse_training_from=arm.get("reuse_training_from", ""),
            )
            for idx, arm in enumerate(arms, start=1)
        ],
        benchmarks=[
            DownstreamBenchmarkSuiteConfig(
                name=item.get("name", f"{item.get('dataset_name', 'miniF2F')}_{item.get('split', 'test')}"),
                dataset_name=item.get("dataset_name", "miniF2F"),
                split=item.get("split", "test"),
                data_path=item.get("data_path", ""),
                manifest_path=item.get("manifest_path", ""),
                task_limit=int(item.get("task_limit", 0)),
                command_template=item.get("command_template", ""),
                timeout_sec=int(item.get("timeout_sec", 7200)),
                pass_k=int(item.get("pass_k", 32)),
                temperature=float(item.get("temperature", 1.0)),
                top_p=float(item.get("top_p", 0.95)),
                max_new_tokens=int(item.get("max_new_tokens", 2048)),
                seed=int(item.get("seed", 7)),
            )
            for item in benchmarks
        ],
    )
