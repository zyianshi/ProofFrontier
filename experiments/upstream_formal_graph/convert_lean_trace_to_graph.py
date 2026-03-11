from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taam.upstream.lean_trace import LeanTraceExtractor


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Lean trace JSON to TAAM graph JSON")
    parser.add_argument("--lean-trace-json", type=str, required=True)
    parser.add_argument("--out-graph-json", type=str, required=True)
    args = parser.parse_args()

    graph = LeanTraceExtractor.from_json(Path(args.lean_trace_json))
    payload = {
        "theorem_id": graph.theorem_id,
        "target_id": graph.target_id,
        "imports": graph.imports,
        "theorem_context": graph.theorem_context,
        "nodes": [
            {
                "id": n.node_id,
                "kind": n.kind,
                "statement": n.statement,
                "lean_statement": n.lean_statement,
                "difficulty": n.difficulty,
                "metadata": n.metadata,
            }
            for n in graph.nodes.values()
        ],
        "edges": [[src, dst] for src, dsts in graph.out_edges.items() for dst in sorted(dsts)],
    }
    out_path = Path(args.out_graph_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved graph json: {out_path.resolve()}")


if __name__ == "__main__":
    main()
