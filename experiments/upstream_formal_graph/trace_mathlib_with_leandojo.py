from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taam.upstream.leandojo_upstream import export_theorem_inventory, load_traced_repo, trace_repo_to_disk


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace Mathlib4 with LeanDojo and export theorem inventory")
    parser.add_argument("--local-repo-path", type=str, default="", help="Local Mathlib4 checkout path")
    parser.add_argument(
        "--repo-url",
        type=str,
        default="https://github.com/leanprover-community/mathlib4",
        help="Mathlib4 repository URL when not using --local-repo-path",
    )
    parser.add_argument("--commit", type=str, default="", help="Commit hash for reproducible tracing")
    parser.add_argument("--traced-repo-root", type=str, required=True, help="Output directory for LeanDojo traced repo")
    parser.add_argument("--load-existing", action="store_true", help="Load existing traced repo instead of tracing")
    parser.add_argument("--no-build-deps", action="store_true", default=False)
    parser.add_argument("--inventory-jsonl", type=str, default="", help="Optional theorem inventory output JSONL")
    parser.add_argument("--allow-non-tactic", action="store_true", default=False)
    parser.add_argument("--min-tactics", type=int, default=1)
    args = parser.parse_args()

    build_deps = False if args.no_build_deps else True
    traced_root = Path(args.traced_repo_root)
    if args.load_existing:
        traced_repo = load_traced_repo(str(traced_root), build_deps=build_deps)
        trace_mode = "load_existing"
    else:
        traced_repo = trace_repo_to_disk(
            repo_url=args.repo_url,
            commit=args.commit,
            local_repo_path=args.local_repo_path,
            dst_dir=str(traced_root),
            build_deps=build_deps,
        )
        trace_mode = "trace_repo"

    inventory_count = 0
    if args.inventory_jsonl:
        inventory_count = export_theorem_inventory(
            traced_repo,
            Path(args.inventory_jsonl),
            require_tactic_proof=not args.allow_non_tactic,
            min_tactics=args.min_tactics,
        )

    summary = {
        "mode": trace_mode,
        "traced_repo_root": str(traced_root),
        "inventory_jsonl": args.inventory_jsonl,
        "inventory_count": inventory_count,
        "build_deps": build_deps,
        "repo_url": args.repo_url,
        "commit": args.commit,
        "local_repo_path": args.local_repo_path,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
