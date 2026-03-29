from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import tools.train_premise_helper as train_premise_helper


class ResolveHelperModelDirTests(unittest.TestCase):
    def test_resolve_helper_model_dir_accepts_local_directory_with_weights(self) -> None:
        with tempfile.TemporaryDirectory(prefix="helper_model_local_") as tmp_dir:
            model_dir = Path(tmp_dir)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "pytorch_model.bin").write_bytes(b"weights")

            resolved = train_premise_helper.resolve_helper_model_dir(str(model_dir))

            self.assertEqual(resolved, model_dir)

    def test_resolve_helper_model_dir_rejects_snapshot_without_weights(self) -> None:
        with tempfile.TemporaryDirectory(prefix="helper_model_snapshot_") as tmp_dir:
            model_dir = Path(tmp_dir)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")

            with mock.patch.object(train_premise_helper, "snapshot_download", return_value=str(model_dir)):
                with self.assertRaises(SystemExit) as ctx:
                    train_premise_helper.resolve_helper_model_dir("microsoft/codebert-base")

            self.assertIn("missing model weights", str(ctx.exception))


class RankingMetricTests(unittest.TestCase):
    def test_compute_ranking_metrics_uses_mrr_recall_hit_and_pairwise_win_rate(self) -> None:
        rows = [
            {"sample_id": "s1", "premise_id": "p1", "label": 1},
            {"sample_id": "s1", "premise_id": "n1", "label": 0},
            {"sample_id": "s1", "premise_id": "n2", "label": 0},
            {"sample_id": "s2", "premise_id": "p2", "label": 1},
            {"sample_id": "s2", "premise_id": "n3", "label": 0},
        ]
        scores = [0.9, 0.8, 0.1, 0.2, 0.7]

        metrics = train_premise_helper.compute_ranking_metrics(rows, scores, top_ks=(1, 2))

        self.assertAlmostEqual(metrics["mrr"], 0.75)
        self.assertAlmostEqual(metrics["hit_at_1"], 0.5)
        self.assertAlmostEqual(metrics["hit_at_2"], 1.0)
        self.assertAlmostEqual(metrics["recall_at_1"], 0.5)
        self.assertAlmostEqual(metrics["recall_at_2"], 1.0)
        self.assertAlmostEqual(metrics["pairwise_win_rate"], 2 / 3)

    def test_is_better_ranking_checkpoint_prefers_mrr_then_recall(self) -> None:
        baseline = {
            "mrr": 0.40,
            "recall_at_8": 0.50,
            "pairwise_win_rate": 0.60,
        }
        improved = {
            "mrr": 0.45,
            "recall_at_8": 0.45,
            "pairwise_win_rate": 0.55,
        }
        tie_break = {
            "mrr": 0.40,
            "recall_at_8": 0.60,
            "pairwise_win_rate": 0.55,
        }

        self.assertTrue(
            train_premise_helper.is_better_ranking_checkpoint(
                improved,
                baseline,
                hint_top_k=8,
            )
        )
        self.assertTrue(
            train_premise_helper.is_better_ranking_checkpoint(
                tie_break,
                baseline,
                hint_top_k=8,
            )
        )
        self.assertFalse(
            train_premise_helper.is_better_ranking_checkpoint(
                baseline,
                improved,
                hint_top_k=8,
            )
        )


if __name__ == "__main__":
    unittest.main()
