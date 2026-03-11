from __future__ import annotations

import argparse
import json
from pathlib import Path


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock prover trainer for downstream pipeline smoke tests")
    parser.add_argument("--train-jsonl", type=str, required=True)
    parser.add_argument("--eval-jsonl", type=str, required=True)
    parser.add_argument("--test-jsonl", type=str, default="")
    parser.add_argument("--output-model-dir", type=str, required=True)
    parser.add_argument("--result-json", type=str, required=True)
    parser.add_argument("--base-model", type=str, default="deepseek-ai/DeepSeek-Prover-V2-7B")
    parser.add_argument("--mode", type=str, default="rl")
    args = parser.parse_args()

    train_count = _count_jsonl(Path(args.train_jsonl))
    eval_count = _count_jsonl(Path(args.eval_jsonl))
    test_count = _count_jsonl(Path(args.test_jsonl)) if args.test_jsonl else 0

    output_model_dir = Path(args.output_model_dir)
    output_model_dir.mkdir(parents=True, exist_ok=True)
    (output_model_dir / "README.mock_model.txt").write_text(
        f"Mock trained model\nbase_model={args.base_model}\nmode={args.mode}\ntrain_count={train_count}\n",
        encoding="utf-8",
    )

    result = {
        "success": True,
        "mode": args.mode,
        "base_model": args.base_model,
        "train_examples": train_count,
        "eval_examples": eval_count,
        "test_examples": test_count,
        "output_model_dir": str(output_model_dir),
    }
    Path(args.result_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TAAM_TRAIN_VERDICT: PASSED")


if __name__ == "__main__":
    main()
