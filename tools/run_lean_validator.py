from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import uuid
from pathlib import Path


def _looks_like_wsl_path(path: str) -> bool:
    path = str(path or '').strip()
    return bool(path) and path.startswith('/')


def _wsl_to_unc(path: str, distro: str) -> Path:
    norm = str(path).replace('/', '\\').lstrip('\\')
    return Path(rf"\\wsl.localhost\{distro}\{norm}")


def _build_command(args: argparse.Namespace) -> str:
    if args.backend_command:
        return args.backend_command
    env_command = os.environ.get('LEAN_VALIDATOR_BACKEND_COMMAND', '').strip()
    if env_command:
        return env_command
    if args.mock_pass:
        return ''
    raise RuntimeError(
        'No Lean validator backend configured. Set LEAN_VALIDATOR_BACKEND_COMMAND '
        'or pass --backend-command, or install Lean and use --project-root / --lean-bin.'
    )


def _write_result(out_json: str, result: dict) -> None:
    Path(out_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


def _run_shell(command: str, timeout_sec: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )


def _run_local_lean(args: argparse.Namespace, lean_text: str) -> dict:
    lean_bin = args.lean_bin or os.environ.get('TAAM_LEAN_BIN', 'lean')
    lake_bin = args.lake_bin or os.environ.get('TAAM_LAKE_BIN', 'lake')
    project_root = args.project_root or os.environ.get('TAAM_LEAN_PROJECT_ROOT', '')
    if not project_root:
        raise RuntimeError('Local Lean validation requires --project-root or TAAM_LEAN_PROJECT_ROOT')

    project = Path(project_root).resolve()
    filename = f'TaamCheck_{uuid.uuid4().hex}.lean'
    target_file = project / filename
    target_file.write_text(lean_text, encoding='utf-8')
    try:
        proc = subprocess.run(
            [lake_bin, 'env', lean_bin, filename],
            capture_output=True,
            text=True,
            timeout=args.timeout_sec,
            cwd=str(project),
        )
        passed = proc.returncode == 0
        return {
            'passed': passed,
            'well_posed': passed,
            'engine': 'local_lean_validator',
            'stdout': proc.stdout,
            'stderr': proc.stderr,
            'returncode': proc.returncode,
        }
    except subprocess.TimeoutExpired:
        return {'passed': False, 'well_posed': False, 'engine': 'local_lean_validator', 'error': 'timeout'}
    finally:
        try:
            target_file.unlink()
        except OSError:
            pass


def _run_wsl_lean(args: argparse.Namespace, lean_text: str) -> dict:
    project_root = args.project_root or os.environ.get('TAAM_LEAN_PROJECT_ROOT', '')
    if not project_root:
        raise RuntimeError('WSL Lean validation requires --project-root to be a WSL path')

    wsl_user = args.wsl_user or os.environ.get('TAAM_WSL_USER', 'xuanxuan_awe')
    wsl_distro = args.wsl_distro or os.environ.get('TAAM_WSL_DISTRO', 'Ubuntu')
    lean_bin = args.lean_bin or os.environ.get('TAAM_LEAN_BIN', 'lean')
    lake_bin = args.lake_bin or os.environ.get('TAAM_LAKE_BIN', 'lake')

    project_unc = _wsl_to_unc(project_root, wsl_distro)
    filename = f'TaamCheck_{uuid.uuid4().hex}.lean'
    target_file = project_unc / filename
    target_file.write_text(lean_text, encoding='utf-8')

    bash_cmd = (
        f'set -euo pipefail; '
        f'cd {shlex.quote(project_root)}; '
        f'. "$HOME/.elan/env"; '
        f'{shlex.quote(lake_bin)} env {shlex.quote(lean_bin)} {shlex.quote(filename)}'
    )
    try:
        proc = subprocess.run(
            ['wsl', '-u', wsl_user, 'bash', '-lc', bash_cmd],
            capture_output=True,
            text=True,
            timeout=args.timeout_sec,
        )
        passed = proc.returncode == 0
        return {
            'passed': passed,
            'well_posed': passed,
            'engine': 'wsl_lean_validator',
            'stdout': proc.stdout,
            'stderr': proc.stderr,
            'returncode': proc.returncode,
            'project_root': project_root,
        }
    except subprocess.TimeoutExpired:
        return {'passed': False, 'well_posed': False, 'engine': 'wsl_lean_validator', 'error': 'timeout'}
    finally:
        try:
            target_file.unlink()
        except OSError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description='Run Lean validator backend for a generated problem')
    parser.add_argument('--lean-file', type=str, required=True)
    parser.add_argument('--out-json', type=str, required=True)
    parser.add_argument('--backend-command', type=str, default='')
    parser.add_argument('--timeout-sec', type=int, default=120)
    parser.add_argument('--mock-pass', action='store_true')
    parser.add_argument('--project-root', type=str, default='')
    parser.add_argument('--lean-bin', type=str, default='')
    parser.add_argument('--lake-bin', type=str, default='')
    parser.add_argument('--use-wsl', action='store_true')
    parser.add_argument('--wsl-user', type=str, default='')
    parser.add_argument('--wsl-distro', type=str, default='Ubuntu')
    parser.add_argument('--disallow-sorry', action='store_true')
    args = parser.parse_args()

    if args.mock_pass:
        result = {'passed': True, 'well_posed': True, 'engine': 'mock_lean_validator'}
        _write_result(args.out_json, result)
        print('TAAM_VALIDATOR_VERDICT: PASSED')
        return

    lean_text = Path(args.lean_file).read_text(encoding='utf-8')
    if args.disallow_sorry and ('sorry' in lean_text or 'admit' in lean_text):
        result = {'passed': False, 'well_posed': False, 'engine': 'lean_validator', 'error': 'found_sorry'}
        _write_result(args.out_json, result)
        print('TAAM_VALIDATOR_VERDICT: ERROR')
        return

    use_wsl = bool(args.use_wsl or (os.name == 'nt' and _looks_like_wsl_path(args.project_root)))
    if args.project_root or args.lean_bin or args.lake_bin:
        result = _run_wsl_lean(args, lean_text) if use_wsl else _run_local_lean(args, lean_text)
        _write_result(args.out_json, result)
        passed = bool(result.get('passed', False))
        print(f"TAAM_VALIDATOR_VERDICT: {'PASSED' if passed else 'ERROR'}")
        return

    command_template = _build_command(args)
    command = command_template.format(
        lean_file=args.lean_file,
        out_json=args.out_json,
        timeout_sec=args.timeout_sec,
    )
    try:
        proc = _run_shell(command, args.timeout_sec)
        passed = proc.returncode == 0
        result = {
            'passed': passed,
            'well_posed': passed,
            'engine': 'external_lean_validator',
            'stdout': proc.stdout,
            'stderr': proc.stderr,
            'returncode': proc.returncode,
        }
    except subprocess.TimeoutExpired:
        passed = False
        result = {'passed': False, 'well_posed': False, 'engine': 'external_lean_validator', 'error': 'timeout'}

    _write_result(args.out_json, result)
    print(f"TAAM_VALIDATOR_VERDICT: {'PASSED' if passed else 'ERROR'}")


if __name__ == '__main__':
    main()
