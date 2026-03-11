from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock benchmark evaluator for downstream pipeline smoke tests")
    parser.add_argument("--model-ref", type=str, required=True)
    parser.add_argument("--benchmark-name", type=str, default="miniF2F")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--out-json", type=str, required=True)
    args = parser.parse_args()

    is_tuned = any(token in args.model_ref.lower() for token in ("trained", "finetuned", "model", "downstream"))
    success_pattern = [False, True, False, True, False] if not is_tuned else [True, True, False, True, True]
    results = []
    for idx, success in enumerate(success_pattern, start=1):
        results.append(
            {
                "task_id": f"{args.benchmark_name}_{args.split}_{idx}",
                "split": args.split,
                "success": success,
                "model_ref": args.model_ref,
            }
        )

    payload = {"benchmark_name": args.benchmark_name, "split": args.split, "results": results}
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TAAM_BENCHMARK_VERDICT: PASSED")


if __name__ == "__main__":
    main()
