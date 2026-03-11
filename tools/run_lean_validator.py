from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


def _build_command(args: argparse.Namespace) -> str:
    if args.backend_command:
        return args.backend_command
    env_command = os.environ.get("LEAN_VALIDATOR_BACKEND_COMMAND", "").strip()
    if env_command:
        return env_command
    if args.mock_pass:
        return ""
    raise RuntimeError(
        "No Lean validator backend configured. Set LEAN_VALIDATOR_BACKEND_COMMAND "
        "or pass --backend-command, or install Lean and use --project-root / --lean-bin."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Lean validator backend for a generated problem")
    parser.add_argument("--lean-file", type=str, required=True)
    parser.add_argument("--out-json", type=str, required=True)
    parser.add_argument("--backend-command", type=str, default="")
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--mock-pass", action="store_true")
    parser.add_argument("--project-root", type=str, default="")
    parser.add_argument("--lean-bin", type=str, default="")
    parser.add_argument("--lake-bin", type=str, default="")
    parser.add_argument("--disallow-sorry", action="store_true")
    args = parser.parse_args()

    if args.mock_pass:
        result = {"passed": True, "engine": "mock_lean_validator"}
        Path(args.out_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print("TAAM_VALIDATOR_VERDICT: PASSED")
        return

    lean_text = Path(args.lean_file).read_text(encoding="utf-8")
    if args.disallow_sorry and ("sorry" in lean_text or "admit" in lean_text):
        result = {
            "passed": False,
            "engine": "local_lean_validator",
            "error": "found_sorry",
        }
        Path(args.out_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print("TAAM_VALIDATOR_VERDICT: ERROR")
        return

    if args.project_root or args.lean_bin:
        lean_bin = args.lean_bin or os.environ.get("TAAM_LEAN_BIN", "lean")
        lake_bin = args.lake_bin or os.environ.get("TAAM_LAKE_BIN", "lake")
        project_root = args.project_root or os.environ.get("TAAM_LEAN_PROJECT_ROOT", "")
        with tempfile.TemporaryDirectory(prefix="taam_lean_check_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_file = tmp_path / "TaamCheck.lean"
            source_file.write_text(lean_text, encoding="utf-8")
            if project_root:
                project = Path(project_root).resolve()
                target_file = project / "TaamCheck.lean"
                target_file.write_text(lean_text, encoding="utf-8")
                command = f'& "{lake_bin}" env "{lean_bin}" "{target_file.name}"'
                workdir = str(project)
            else:
                command = f'& "{lean_bin}" "{source_file}"'
                workdir = str(tmp_path)
            try:
                proc = subprocess.run(
                    ["powershell.exe", "-NoProfile", "-Command", command],
                    capture_output=True,
                    text=True,
                    timeout=args.timeout_sec,
                    cwd=workdir,
                )
                passed = proc.returncode == 0
                result = {
                    "passed": passed,
                    "engine": "local_lean_validator",
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                    "returncode": proc.returncode,
                }
            except subprocess.TimeoutExpired:
                passed = False
                result = {"passed": False, "engine": "local_lean_validator", "error": "timeout"}
            Path(args.out_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"TAAM_VALIDATOR_VERDICT: {'PASSED' if passed else 'ERROR'}")
            return

    command_template = _build_command(args)
    command = command_template.format(
        lean_file=args.lean_file,
        out_json=args.out_json,
        timeout_sec=args.timeout_sec,
    )
    try:
        proc = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            timeout=args.timeout_sec,
        )
        passed = proc.returncode == 0
        result = {
            "passed": passed,
            "engine": "external_lean_validator",
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        passed = False
        result = {"passed": False, "engine": "external_lean_validator", "error": "timeout"}

    Path(args.out_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAAM_VALIDATOR_VERDICT: {'PASSED' if passed else 'ERROR'}")


if __name__ == "__main__":
    main()
