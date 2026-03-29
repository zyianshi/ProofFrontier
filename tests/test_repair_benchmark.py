from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from taam.downstream.repair_benchmark import (
    RepairBenchmarkTask,
    allocate_repair_attempt_budgets,
    merge_repair_benchmark_payloads,
    partition_repair_tasks,
    run_repair_task,
)


class RepairBenchmarkTests(unittest.TestCase):
    def test_partition_repair_tasks_round_robins_evenly(self) -> None:
        tasks = [
            RepairBenchmarkTask(
                task_id=f"task::{idx}",
                theorem_name=f"demo_{idx}",
                split="test",
                source_file="demo.lean",
                lean_code="import Mathlib\n\ntheorem demo : True := by\n  trivial\n",
                dataset="hard-sample-replay",
            )
            for idx in range(5)
        ]

        shards = partition_repair_tasks(tasks, shard_count=2)

        self.assertEqual([[task.task_id for task in shard] for shard in shards], [
            ["task::0", "task::2", "task::4"],
            ["task::1", "task::3"],
        ])

    def test_merge_repair_benchmark_payloads_preserves_totals_and_order(self) -> None:
        payload_a = {
            "benchmark_name": "hard_sample_replay",
            "split": "test",
            "tasks": 2,
            "successes": 1,
            "pass_k": 32,
            "mode": "repair_helper_budget32",
            "budget_schedule": "24/8",
            "hint_source": "learned",
            "results": [
                {"task_id": "task::0", "success": True},
                {"task_id": "task::2", "success": False},
            ],
        }
        payload_b = {
            "benchmark_name": "hard_sample_replay",
            "split": "test",
            "tasks": 2,
            "successes": 1,
            "pass_k": 32,
            "mode": "repair_helper_budget32",
            "budget_schedule": "24/8",
            "hint_source": "learned",
            "results": [
                {"task_id": "task::1", "success": False},
                {"task_id": "task::3", "success": True},
            ],
        }

        merged = merge_repair_benchmark_payloads(
            [payload_a, payload_b],
            ordered_task_ids=["task::0", "task::1", "task::2", "task::3"],
        )

        self.assertEqual(merged["tasks"], 4)
        self.assertEqual(merged["successes"], 2)
        self.assertEqual(
            [row["task_id"] for row in merged["results"]],
            ["task::0", "task::1", "task::2", "task::3"],
        )

    def test_allocate_repair_attempt_budgets(self) -> None:
        self.assertEqual(
            allocate_repair_attempt_budgets(mode="repair_helper_budget32", pass_k=32, schedule_name="24/8"),
            {"vanilla_attempts": 24, "repair_attempts": 8},
        )
        self.assertEqual(
            allocate_repair_attempt_budgets(mode="repair_helper_extra_budget", pass_k=32, schedule_name="24/8"),
            {"vanilla_attempts": 32, "repair_attempts": 32},
        )

    def test_run_repair_task_solves_only_after_hint_injection(self) -> None:
        with tempfile.TemporaryDirectory(prefix="repair_benchmark_test_") as tmp_dir:
            tmp = Path(tmp_dir)
            solver_script = tmp / "mock_solver.py"
            solver_script.write_text(
                textwrap.dedent(
                    """
                    import json
                    import sys
                    from pathlib import Path

                    lean_file = Path(sys.argv[1])
                    out_json = Path(sys.argv[2])
                    text = lean_file.read_text(encoding="utf-8")
                    solved = ("needed_lemma : True" in text)
                    out_json.write_text(json.dumps({"solved": solved}), encoding="utf-8")
                    raise SystemExit(0 if solved else 1)
                    """
                ),
                encoding="utf-8",
            )
            task = RepairBenchmarkTask(
                task_id="hard::demo",
                theorem_name="demo",
                split="test",
                source_file="demo.lean",
                lean_code="import Mathlib\n\ntheorem demo : True := by\n  sorry\n",
                dataset="hard-sample-replay",
                language="lean4",
                metadata={},
            )
            helper_hits = [{"premise_id": "needed_lemma", "target_statement": "True"}]
            result = run_repair_task(
                task=task,
                command_template=f'python "{solver_script}" "{{lean_file}}" "{{result_json}}"',
                model_ref="frozen-base",
                server_url="",
                timeout_sec=30,
                temperature=0.0,
                top_p=1.0,
                max_new_tokens=32,
                seed=7,
                vanilla_attempts=1,
                repair_attempts=1,
                helper_hits=helper_hits,
            )

            self.assertFalse(result["vanilla_success"])
            self.assertTrue(result["repair_success"])
            self.assertTrue(result["success"])
            self.assertEqual(result["successful_stage"], "repair")

    def test_run_repair_task_keeps_vanilla_success_without_repair(self) -> None:
        with tempfile.TemporaryDirectory(prefix="repair_benchmark_test_") as tmp_dir:
            tmp = Path(tmp_dir)
            solver_script = tmp / "mock_solver.py"
            solver_script.write_text(
                textwrap.dedent(
                    """
                    import json
                    import sys
                    from pathlib import Path

                    out_json = Path(sys.argv[2])
                    out_json.write_text(json.dumps({"solved": True}), encoding="utf-8")
                    raise SystemExit(0)
                    """
                ),
                encoding="utf-8",
            )
            task = RepairBenchmarkTask(
                task_id="easy::demo",
                theorem_name="demo",
                split="test",
                source_file="demo.lean",
                lean_code="import Mathlib\n\ntheorem demo : True := by\n  sorry\n",
                dataset="hard-sample-replay",
                language="lean4",
                metadata={},
            )
            result = run_repair_task(
                task=task,
                command_template=f'python "{solver_script}" "{{lean_file}}" "{{result_json}}"',
                model_ref="frozen-base",
                server_url="",
                timeout_sec=30,
                temperature=0.0,
                top_p=1.0,
                max_new_tokens=32,
                seed=7,
                vanilla_attempts=1,
                repair_attempts=1,
                helper_hits=[{"premise_id": "needed_lemma", "target_statement": "True"}],
            )

            self.assertTrue(result["vanilla_success"])
            self.assertFalse(result["repair_attempt_results"])
            self.assertEqual(result["successful_stage"], "vanilla")

    def test_repair_benchmark_tool_supports_oracle_hints_without_helper_weights(self) -> None:
        with tempfile.TemporaryDirectory(prefix="repair_benchmark_tool_") as tmp_dir:
            tmp = Path(tmp_dir)
            solver_script = tmp / "mock_solver.py"
            solver_script.write_text(
                textwrap.dedent(
                    """
                    import json
                    import sys
                    from pathlib import Path

                    lean_file = Path(sys.argv[1])
                    out_json = Path(sys.argv[2])
                    text = lean_file.read_text(encoding="utf-8")
                    solved = "needed_lemma : True" in text
                    out_json.write_text(json.dumps({"solved": solved}), encoding="utf-8")
                    raise SystemExit(0 if solved else 1)
                    """
                ),
                encoding="utf-8",
            )
            manifest_path = tmp / "manifest.jsonl"
            manifest_row = {
                "task_id": "hard::demo",
                "theorem_name": "demo",
                "split": "test",
                "source_file": "demo.lean",
                "lean_code": "import Mathlib\n\ntheorem demo : True := by\n  sorry\n",
                "dataset": "hard_sample_replay",
                "language": "lean4",
                "metadata": {"positive_premises": ["needed_lemma"]},
            }
            manifest_path.write_text(json.dumps(manifest_row) + "\n", encoding="utf-8")
            inventory_path = tmp / "inventory.jsonl"
            inventory_path.write_text(
                json.dumps(
                    {
                        "theorem_id": "needed_lemma",
                        "target_statement": "True",
                        "file_path": "Demo.lean",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            out_json = tmp / "result.json"

            proc = subprocess.run(
                [
                    sys.executable,
                    "tools/run_repair_augmented_benchmark.py",
                    "--benchmark-name",
                    "hard_sample_replay",
                    "--manifest-jsonl",
                    str(manifest_path),
                    "--model-ref",
                    "frozen-base",
                    "--premise-inventory-jsonl",
                    str(inventory_path),
                    "--command-template",
                    f'python "{solver_script}" "{{lean_file}}" "{{result_json}}"',
                    "--out-json",
                    str(out_json),
                    "--pass-k",
                    "2",
                    "--vanilla-attempts",
                    "1",
                    "--repair-attempts",
                    "1",
                    "--hint-source",
                    "oracle",
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["successes"], 1)
            self.assertEqual(payload["hint_source"], "oracle")
            self.assertEqual(payload["results"][0]["retrieved_premises"], ["needed_lemma"])


if __name__ == "__main__":
    unittest.main()
