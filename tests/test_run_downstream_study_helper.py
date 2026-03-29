from __future__ import annotations

import json
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


class DownstreamRepairHelperStudyTests(unittest.TestCase):
    def test_helper_three_arm_study_writes_summary_and_csvs(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="repair_helper_study_") as tmp_dir:
            tmp = Path(tmp_dir)
            samples_root = tmp / "samples"
            samples_root.mkdir(parents=True, exist_ok=True)
            for idx in range(8):
                sample = {
                    "theorem_id": f"Demo.theorem_{idx}",
                    "target_id": f"Demo.theorem_{idx}",
                    "failed_on": "Missing.lemma",
                    "hidden_nodes": ["Missing.lemma"],
                    "visible_nodes": ["Visible.one"],
                    "lean_problem": "import Mathlib\n\ntheorem demo : True := by\n  sorry\n",
                    "full_lean_problem": "import Mathlib\n\ntheorem demo : True := by\n  trivial\n",
                    "proof_completion": "by\n  trivial",
                    "well_posed": True,
                }
                (samples_root / f"sample_{idx}.json").write_text(json.dumps(sample), encoding="utf-8")

            inventory_jsonl = tmp / "inventory.jsonl"
            with inventory_jsonl.open("w", encoding="utf-8") as f:
                for idx in range(4):
                    row = {
                        "theorem_id": f"Premise.{idx}",
                        "target_statement": f"goal_{idx} = goal_{idx}",
                        "file_path": f"Mathlib/Premise{idx}.lean",
                    }
                    f.write(json.dumps(row) + "\n")

            trainer_script = tmp / "mock_helper_train.py"
            trainer_script.write_text(
                textwrap.dedent(
                    """
                    import argparse, json
                    from pathlib import Path

                    parser = argparse.ArgumentParser()
                    parser.add_argument("--train-jsonl", required=True)
                    parser.add_argument("--eval-jsonl", required=True)
                    parser.add_argument("--test-jsonl", required=True)
                    parser.add_argument("--inventory-jsonl", required=True)
                    parser.add_argument("--output-model-dir", required=True)
                    parser.add_argument("--result-json", required=True)
                    parser.add_argument("--mode", default="")
                    args = parser.parse_args()

                    out_dir = Path(args.output_model_dir)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    (out_dir / "helper.mock").write_text("ok", encoding="utf-8")
                    Path(args.result_json).write_text(
                        json.dumps(
                            {
                                "success": True,
                                "output_model_dir": str(out_dir),
                                "mode": args.mode,
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    """
                ),
                encoding="utf-8",
            )

            def _write_manifest(path: Path, benchmark_name: str) -> None:
                rows = []
                for idx in range(2):
                    rows.append(
                        {
                            "task_id": f"{benchmark_name}::{idx}",
                            "theorem_name": f"{benchmark_name}_{idx}",
                            "split": "test",
                            "source_file": f"{benchmark_name}_{idx}.lean",
                            "lean_code": "import Mathlib\n\ntheorem demo : True := by\n  sorry\n",
                            "dataset": benchmark_name,
                            "language": "lean4",
                            "metadata": {},
                        }
                    )
                with path.open("w", encoding="utf-8") as f:
                    for row in rows:
                        f.write(json.dumps(row) + "\n")

            mini_manifest = tmp / "mini_manifest.jsonl"
            proof_manifest = tmp / "proof_manifest.jsonl"
            _write_manifest(mini_manifest, "miniF2F")
            _write_manifest(proof_manifest, "ProofNet")

            out_dir = tmp / "study_out"
            config = {
                "name": "repair_helper_study_smoke",
                "out_dir": str(out_dir),
                "premise_helper": {
                    "inventory_jsonl": str(inventory_jsonl),
                    "dataset": {
                        "source_type": "hard_samples",
                        "samples_root": str(samples_root),
                        "sample_glob": "*.json",
                        "format": "helper",
                        "only_well_posed": True,
                        "require_proof_completion": False,
                        "train_ratio": 0.5,
                        "val_ratio": 0.25,
                        "test_ratio": 0.25,
                        "seed": 7,
                    },
                    "bm25_candidate_count": 8,
                    "rerank_top_n": 4,
                    "hint_top_k": 2,
                    "budget_schedule_candidates": ["24/8", "16/16", "8/24"],
                },
                "arms": [
                    {
                        "name": "deepseek_vanilla",
                        "mode": "vanilla",
                        "base_model": "deepseek-ai/DeepSeek-Prover-V2-7B",
                        "benchmark_model_ref": "deepseek-ai/DeepSeek-Prover-V2-7B",
                        "training_enabled": False,
                    },
                    {
                        "name": "deepseek_repair_helper_budget32",
                        "mode": "repair_helper_budget32",
                        "base_model": "deepseek-ai/DeepSeek-Prover-V2-7B",
                        "training_enabled": True,
                        "training_command_template": f'python "{trainer_script}" --train-jsonl "{{train_jsonl}}" --eval-jsonl "{{eval_jsonl}}" --test-jsonl "{{test_jsonl}}" --inventory-jsonl "{{inventory_jsonl}}" --output-model-dir "{{output_model_dir}}" --result-json "{{result_json}}" --mode "{{mode}}"',
                        "compare_to": "deepseek_vanilla",
                    },
                    {
                        "name": "deepseek_repair_helper_extra_budget",
                        "mode": "repair_helper_extra_budget",
                        "base_model": "deepseek-ai/DeepSeek-Prover-V2-7B",
                        "training_enabled": False,
                        "reuse_training_from": "deepseek_repair_helper_budget32",
                        "compare_to": "deepseek_vanilla",
                    },
                ],
                "benchmarks": [
                    {
                        "name": "miniF2F_test",
                        "dataset_name": "miniF2F",
                        "split": "test",
                        "manifest_path": str(mini_manifest),
                        "command_template": "python tools/mock_benchmark_eval.py --model-ref \"{model_ref}\" --benchmark-name \"{benchmark_name}\" --split \"{split}\" --out-json \"{out_json}\"",
                        "timeout_sec": 60,
                        "pass_k": 32,
                    },
                    {
                        "name": "ProofNet_test",
                        "dataset_name": "ProofNet",
                        "split": "test",
                        "manifest_path": str(proof_manifest),
                        "command_template": "python tools/mock_benchmark_eval.py --model-ref \"{model_ref}\" --benchmark-name \"{benchmark_name}\" --split \"{split}\" --out-json \"{out_json}\"",
                        "timeout_sec": 60,
                        "pass_k": 32,
                    },
                ],
            }
            config_path = tmp / "config.json"
            config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

            subprocess.run(
                ["python", "experiments/downstream_prover_eval/run_downstream_study.py", "--config", str(config_path)],
                cwd=repo_root,
                check=True,
            )

            summary = json.loads((out_dir / "downstream_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(summary["main_results"]), 6)
            self.assertEqual(len(summary["pairwise_comparisons"]), 4)
            self.assertIn("hard_sample_replay_test", summary["targeted_results"])
            self.assertEqual(len(summary["targeted_results"]["hard_sample_replay_test"]["by_arm"]), 3)
            self.assertTrue((out_dir / "main_results.csv").exists())
            self.assertTrue((out_dir / "pairwise_comparisons.csv").exists())


if __name__ == "__main__":
    unittest.main()
