from __future__ import annotations

import subprocess
import tempfile
from os import environ
from pathlib import Path
from typing import Any, Dict, Optional

from ..external_tools import parse_json_result, parse_verdict_from_text, run_command_template
from .formalization import build_lean4_problem_from_state
from ..types import ProblemState, TAAMGraph


class Solver:
    """
    Phase 3 solver/prover interface.
    """

    def solve(self, state: ProblemState, graph: TAAMGraph) -> bool:
        raise NotImplementedError


class DeepSeekProverSolver(Solver):
    """
    Prover-based solver adapter.
    It creates a Lean4 problem from current visible graph state, then calls an external prover command.

    command_template placeholders:
    - {lean_file}
    - {result_json}
    - {timeout_sec}
    """

    def __init__(self, command_template: str, timeout_sec: int = 180):
        if "{lean_file}" not in command_template:
            raise ValueError("command_template must include {lean_file}")
        self.command_template = command_template
        self.timeout_sec = timeout_sec

    def solve(self, state: ProblemState, graph: TAAMGraph) -> bool:
        lean_problem = build_lean4_problem_from_state(state)
        with tempfile.TemporaryDirectory(prefix="taam_prover_") as tmp_dir:
            tmp = Path(tmp_dir)
            lean_file = tmp / "problem.lean"
            result_json = tmp / "result.json"
            lean_file.write_text(lean_problem, encoding="utf-8")

            try:
                proc = run_command_template(
                    self.command_template,
                    {
                        "lean_file": str(lean_file),
                        "result_json": str(result_json),
                        "timeout_sec": str(self.timeout_sec),
                    },
                    timeout_sec=self.timeout_sec,
                )
            except subprocess.TimeoutExpired:
                return False

            json_verdict = parse_json_result(result_json, ("solved", "proved", "success"))
            if json_verdict is not None:
                return json_verdict
            parsed = parse_verdict_from_text(f"{proc.stdout}\n{proc.stderr}", "TAAM_PROVER_VERDICT")
            if parsed is not None:
                return parsed
            return proc.returncode == 0


def create_solver(
    solver_type: str,
    threshold: float,
    seed: int,
    solver_config: Optional[Dict[str, Any]] = None,
) -> Solver:
    cfg = solver_config or {}
    st = solver_type.lower().strip()
    _ = threshold
    _ = seed
    if st == "deepseek_prover":
        command_template = str(
            cfg.get(
                "command_template",
                environ.get(
                    "TAAM_DEEPSEEK_PROVER_COMMAND",
                    'python tools/run_deepseek_prover.py --lean-file "{lean_file}" --out-json "{result_json}"',
                ),
            )
        )
        return DeepSeekProverSolver(
            command_template=command_template,
            timeout_sec=int(cfg.get("timeout_sec", 180)),
        )
    raise ValueError(f"Unknown solver_type: {solver_type}")
