from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _run(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap a local Lean + Mathlib project for TAAM")
    parser.add_argument("--project-root", type=str, default="lean_project")
    parser.add_argument("--project-name", type=str, default="TaamLean")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    if not (root / "lakefile.toml").exists() and not (root / "lakefile.lean").exists():
        _run(["lake", "init", args.project_name], root)
    _run(["lake", "add", "mathlib"], root)
    _run(["lake", "exe", "cache", "get"], root)


if __name__ == "__main__":
    main()
