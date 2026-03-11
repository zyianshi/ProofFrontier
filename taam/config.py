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
    out_dir: str = "artifacts/midstream_taam_generation/runs"
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
    lean_trace_json: str = ""
    out_root: str = "artifacts/midstream_taam_generation/sweeps"
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
    samples_root: str = "artifacts/midstream_taam_generation/runs"
    sample_glob: str = "**/*_hard_sample.json"
    format: str = "rl"
    only_well_posed: bool = True
    require_proof_completion: bool = False
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 7


@dataclass
class DownstreamTrainingConfig:
    enabled: bool = False
    mode: str = "sft"
    base_model: str = "deepseek-ai/DeepSeek-Prover-V2-7B"
    output_model_dir: str = "artifacts/downstream_prover_eval/model"
    command_template: str = ""
    timeout_sec: int = 7200


@dataclass
class DownstreamBenchmarkConfig:
    enabled: bool = False
    dataset_name: str = "miniF2F"
    split: str = "test"
    data_path: str = ""
    manifest_path: str = ""
    task_limit: int = 0
    base_model_ref: str = "deepseek-ai/DeepSeek-Prover-V2-7B"
    tuned_model_ref: str = ""
    base_command_template: str = ""
    tuned_command_template: str = ""
    timeout_sec: int = 7200


@dataclass
class DownstreamConfig:
    name: str = "downstream"
    out_dir: str = "artifacts/downstream_prover_eval"
    dataset: DownstreamDatasetConfig = field(default_factory=DownstreamDatasetConfig)
    training: DownstreamTrainingConfig = field(default_factory=DownstreamTrainingConfig)
    benchmark: DownstreamBenchmarkConfig = field(default_factory=DownstreamBenchmarkConfig)


def load_experiment_config(path: Path) -> ExperimentConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    return ExperimentConfig(
        name=data.get("name", "default"),
        graph_json=data.get("graph_json", ""),
        lean_trace_json=data.get("lean_trace_json", ""),
        out_dir=data.get("out_dir", "artifacts/midstream_taam_generation/runs"),
        capability_threshold=float(data.get("capability_threshold", 0.0)),
        seed=int(data.get("seed", 7)),
        solver_type=data.get("solver_type", "deepseek_prover"),
        solver_config=dict(data.get("solver_config", {})),
        validation_config=dict(data.get("validation_config", {})),
        masking_strategy=data.get("masking_strategy", "greedy_topology"),
        masking_config=dict(data.get("masking_config", {})),
    )


def load_sweep_config(path: Path) -> SweepConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    return SweepConfig(
        name=data.get("name", "sweep"),
        graph_json=data.get("graph_json", ""),
        lean_trace_json=data.get("lean_trace_json", ""),
        out_root=data.get("out_root", "artifacts/midstream_taam_generation/sweeps"),
        capability_thresholds=[float(x) for x in data.get("capability_thresholds", [0.0])],
        seeds=[int(x) for x in data.get("seeds", [7, 13, 42])],
        solver_type=data.get("solver_type", "deepseek_prover"),
        solver_config=dict(data.get("solver_config", {})),
        validation_config=dict(data.get("validation_config", {})),
        masking_strategy=data.get("masking_strategy", "greedy_topology"),
        masking_config=dict(data.get("masking_config", {})),
    )


def load_downstream_config(path: Path) -> DownstreamConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    dataset = data.get("dataset", {})
    training = data.get("training", {})
    benchmark = data.get("benchmark", {})
    return DownstreamConfig(
        name=data.get("name", "downstream"),
        out_dir=data.get("out_dir", "artifacts/downstream_prover_eval"),
        dataset=DownstreamDatasetConfig(
            samples_root=dataset.get("samples_root", "artifacts/midstream_taam_generation/runs"),
            sample_glob=dataset.get("sample_glob", "**/*_hard_sample.json"),
            format=dataset.get("format", "rl"),
            only_well_posed=bool(dataset.get("only_well_posed", True)),
            require_proof_completion=bool(dataset.get("require_proof_completion", False)),
            train_ratio=float(dataset.get("train_ratio", 0.8)),
            val_ratio=float(dataset.get("val_ratio", 0.1)),
            test_ratio=float(dataset.get("test_ratio", 0.1)),
            seed=int(dataset.get("seed", 7)),
        ),
        training=DownstreamTrainingConfig(
            enabled=bool(training.get("enabled", False)),
            mode=training.get("mode", "sft"),
            base_model=training.get("base_model", "deepseek-ai/DeepSeek-Prover-V2-7B"),
            output_model_dir=training.get("output_model_dir", "artifacts/downstream_prover_eval/model"),
            command_template=training.get("command_template", ""),
            timeout_sec=int(training.get("timeout_sec", 7200)),
        ),
        benchmark=DownstreamBenchmarkConfig(
            enabled=bool(benchmark.get("enabled", False)),
            dataset_name=benchmark.get("dataset_name", "miniF2F"),
            split=benchmark.get("split", "test"),
            data_path=benchmark.get("data_path", ""),
            manifest_path=benchmark.get("manifest_path", ""),
            task_limit=int(benchmark.get("task_limit", 0)),
            base_model_ref=benchmark.get("base_model_ref", "deepseek-ai/DeepSeek-Prover-V2-7B"),
            tuned_model_ref=benchmark.get("tuned_model_ref", ""),
            base_command_template=benchmark.get("base_command_template", ""),
            tuned_command_template=benchmark.get("tuned_command_template", ""),
            timeout_sec=int(benchmark.get("timeout_sec", 7200)),
        ),
    )
