from __future__ import annotations

import subprocess
import tempfile
from os import environ
from pathlib import Path
from typing import Any, Dict, Optional

from ..external_tools import parse_json_result, parse_verdict_from_text, run_command_template
from ..types import HardProblemSample, TAAMGraph


class SafetyNetValidator:
    """
    Lean-side well-posedness checker.
    It type-checks the generated Lean problem through an external command.
    """

    @staticmethod
    def validate(
        sample: HardProblemSample,
        _graph: TAAMGraph,
        validation_config: Optional[Dict[str, Any]] = None,
    ) -> Optional[bool]:
        cfg = validation_config or {}
        enabled = bool(cfg.get("enabled", True))
        if not enabled:
            return None

        command_template = str(
            cfg.get(
                "command_template",
                environ.get(
                    "TAAM_LEAN_VALIDATOR_COMMAND",
                    'python tools/run_lean_validator.py --lean-file "{lean_file}" --out-json "{result_json}"',
                ),
            )
        )
        timeout_sec = int(cfg.get("timeout_sec", 120))

        with tempfile.TemporaryDirectory(prefix="taam_validator_") as tmp_dir:
            tmp = Path(tmp_dir)
            lean_file = tmp / "problem.lean"
            result_json = tmp / "result.json"
            lean_file.write_text(sample.lean_problem, encoding="utf-8")
            try:
                proc = run_command_template(
                    command_template,
                    {
                        "lean_file": str(lean_file),
                        "result_json": str(result_json),
                        "timeout_sec": str(timeout_sec),
                    },
                    timeout_sec=timeout_sec,
                )
            except subprocess.TimeoutExpired:
                return False

            json_verdict = parse_json_result(result_json, ("passed", "well_posed", "success"))
            if json_verdict is not None:
                return json_verdict
            parsed = parse_verdict_from_text(f"{proc.stdout}\n{proc.stderr}", "TAAM_VALIDATOR_VERDICT")
            if parsed is not None:
                return parsed
            return proc.returncode == 0
