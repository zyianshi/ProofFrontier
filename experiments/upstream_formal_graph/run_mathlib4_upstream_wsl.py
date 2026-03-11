from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run_wsl_python(
    *,
    wsl_user: str,
    wsl_project_root: str,
    wsl_python_bin: str,
    script_rel_path: str,
    args: List[str],
    timeout_sec: int,
) -> None:
    quoted_root = shlex.quote(wsl_project_root)
    quoted_python = shlex.quote(wsl_python_bin)
    quoted_script = shlex.quote(script_rel_path)
    quoted_args = " ".join(shlex.quote(str(arg)) for arg in args)
    bash_cmd = (
        f"set -euo pipefail; cd {quoted_root}; "
        f". \"$HOME/.elan/env\"; "
        f"{quoted_python} {quoted_script} {quoted_args}"
    )
    proc = subprocess.run(
        ["wsl", "-u", wsl_user, "bash", "-lc", bash_cmd],
        text=True,
        timeout=timeout_sec,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the real Mathlib4 -> LeanDojo -> TAAM upstream flow inside WSL")
    parser.add_argument("--config", type=str, default="configs/upstream_formal_graph/mathlib4_leandojo_upstream.json")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    wsl_user = str(cfg.get("wsl_user", "xuanxuan_awe"))
    wsl_project_root = str(cfg.get("wsl_project_root", "/home/xuanxuan_awe/GenAI"))
    wsl_python_bin = str(cfg.get("wsl_python_bin", "/home/xuanxuan_awe/miniconda3/envs/work/bin/python"))
    timeout_sec = int(cfg.get("timeout_sec", 7200))

    trace_args: List[str] = [
        "--repo-url",
        str(cfg["repo_url"]),
        "--commit",
        str(cfg["commit"]),
        "--traced-repo-root",
        str(cfg["traced_repo_root"]),
        "--inventory-jsonl",
        str(cfg["inventory_jsonl"]),
        "--min-tactics",
        str(int(cfg.get("min_tactics", 1))),
    ]
    if cfg.get("local_repo_path"):
        trace_args.extend(["--local-repo-path", str(cfg["local_repo_path"])])
    if bool(cfg.get("load_existing", False)):
        trace_args.append("--load-existing")
    if bool(cfg.get("no_build_deps", False)):
        trace_args.append("--no-build-deps")
    if bool(cfg.get("allow_non_tactic", False)):
        trace_args.append("--allow-non-tactic")

    build_args: List[str] = [
        "--traced-repo-root",
        str(cfg["traced_repo_root"]),
        "--inventory-jsonl",
        str(cfg["inventory_jsonl"]),
        "--out-dir",
        str(cfg["graph_out_dir"]),
        "--min-tactics",
        str(int(cfg.get("min_tactics", 1))),
        "--limit",
        str(int(cfg.get("limit", 10))),
        "--max-depth",
        str(int(cfg.get("max_depth", 2))),
        "--max-nodes",
        str(int(cfg.get("max_nodes", 128))),
    ]
    if bool(cfg.get("allow_non_tactic", False)):
        build_args.append("--allow-non-tactic")
    if bool(cfg.get("no_build_deps", False)):
        build_args.append("--no-build-deps")
    for prefix in cfg.get("module_prefixes", []):
        build_args.extend(["--module-prefix", str(prefix)])

    _run_wsl_python(
        wsl_user=wsl_user,
        wsl_project_root=wsl_project_root,
        wsl_python_bin=wsl_python_bin,
        script_rel_path="experiments/upstream_formal_graph/trace_mathlib_with_leandojo.py",
        args=trace_args,
        timeout_sec=timeout_sec,
    )
    _run_wsl_python(
        wsl_user=wsl_user,
        wsl_project_root=wsl_project_root,
        wsl_python_bin=wsl_python_bin,
        script_rel_path="experiments/upstream_formal_graph/build_mathlib_trace_corpus.py",
        args=build_args,
        timeout_sec=timeout_sec,
    )

    summary = {
        "config": str(Path(args.config)),
        "traced_repo_root": str(cfg["traced_repo_root"]),
        "inventory_jsonl": str(cfg["inventory_jsonl"]),
        "graph_out_dir": str(cfg["graph_out_dir"]),
        "limit": int(cfg.get("limit", 10)),
        "wsl_python_bin": wsl_python_bin,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
