from __future__ import annotations

import random
import unittest

from taam.downstream.repair_helper import (
    BM25Index,
    build_premise_corpus,
    build_pairwise_ranking_examples,
    hard_sample_to_helper_record,
    inject_premise_hints,
    is_algebra_like_task_name,
    select_hint_premises,
    select_best_budget_schedule,
)


class RepairHelperDataTests(unittest.TestCase):
    def test_hard_sample_to_helper_record_uses_failed_and_hidden_as_positives(self) -> None:
        sample = {
            "theorem_id": "Foo.bar",
            "target_id": "Foo.bar",
            "failed_on": "Missing.lemma",
            "hidden_nodes": ["Hidden.one", "Missing.lemma"],
            "visible_nodes": ["Visible.one", "Visible.two"],
            "lean_problem": "import Mathlib\n\ntheorem demo : True := by\n  sorry\n",
            "well_posed": True,
            "_source_path": "/tmp/sample.json",
        }

        record = hard_sample_to_helper_record(sample)

        self.assertEqual(record["query"], sample["lean_problem"])
        self.assertEqual(record["positive_premises"], ["Hidden.one", "Missing.lemma"])
        self.assertEqual(record["negative_premises"], ["Visible.one", "Visible.two"])

    def test_build_premise_corpus_keeps_theorem_identity_and_target(self) -> None:
        rows = [
            {
                "theorem_id": "Mathlib.Algebra.Polynomial.foo",
                "target_statement": "x = x",
                "file_path": "Mathlib/Algebra/Polynomial/Foo.lean",
            },
            {
                "theorem_id": "Mathlib.Algebra.Polynomial.bar",
                "target_statement": "y = y",
                "file_path": "Mathlib/Algebra/Polynomial/Bar.lean",
            },
        ]

        corpus = build_premise_corpus(rows)

        self.assertEqual(len(corpus), 2)
        self.assertEqual(corpus[0]["premise_id"], "Mathlib.Algebra.Polynomial.foo")
        self.assertEqual(corpus[0]["target_statement"], "x = x")
        self.assertEqual(corpus[0]["source_file_path"], "Mathlib/Algebra/Polynomial/Foo.lean")

    def test_inject_premise_hints_uses_comment_block_without_synthetic_declarations(self) -> None:
        lean_problem = "import Mathlib\n\nnamespace Demo\n\ntheorem demo : True := by\n  sorry\n"
        premises = [
            {"premise_id": "Foo.bar", "target_statement": "True"},
            {"premise_id": "Baz.qux", "target_statement": "x = x"},
        ]

        hinted = inject_premise_hints(lean_problem, premises)

        self.assertIn("/- TAAM premise hints:", hinted)
        self.assertIn("Foo.bar : True", hinted)
        self.assertIn("Baz.qux : x = x", hinted)
        self.assertNotIn("theorem Foo.bar", hinted)
        self.assertNotIn("lemma Baz.qux", hinted)

    def test_select_best_budget_schedule_prefers_highest_validation_success(self) -> None:
        scores = {
            "24/8": {"successes": 3},
            "16/16": {"successes": 7},
            "8/24": {"successes": 5},
        }

        chosen = select_best_budget_schedule(
            schedule_candidates=["24/8", "16/16", "8/24"],
            schedule_scores=scores,
        )

        self.assertEqual(chosen, "16/16")

    def test_is_algebra_like_task_name_uses_name_heuristics(self) -> None:
        self.assertTrue(is_algebra_like_task_name("mathd_algebra_182"))
        self.assertTrue(is_algebra_like_task_name("polynomial_degree_case"))
        self.assertFalse(is_algebra_like_task_name("exercise_1_19a"))

    def test_select_hint_premises_oracle_uses_positive_premises(self) -> None:
        corpus = [
            {"premise_id": "Hidden.one", "target_statement": "x = x", "text": "Hidden.one : x = x", "tokens": ["hidden", "one", "x"]},
            {"premise_id": "Missing.lemma", "target_statement": "True", "text": "Missing.lemma : True", "tokens": ["missing", "lemma", "true"]},
            {"premise_id": "Visible.one", "target_statement": "False", "text": "Visible.one : False", "tokens": ["visible", "one", "false"]},
        ]

        hits = select_hint_premises(
            query="theorem demo : True := by sorry",
            task_metadata={"positive_premises": ["Hidden.one", "Missing.lemma"]},
            corpus=corpus,
            bm25=BM25Index(corpus),
            reranker=None,
            hint_source="oracle",
            bm25_candidate_count=8,
            rerank_top_n=4,
            hint_top_k=2,
            rng=random.Random(7),
        )

        self.assertEqual([row["premise_id"] for row in hits], ["Hidden.one", "Missing.lemma"])

    def test_select_hint_premises_random_samples_from_corpus(self) -> None:
        corpus = [
            {"premise_id": "Premise.A", "target_statement": "A", "text": "Premise.A : A", "tokens": ["premise", "a"]},
            {"premise_id": "Premise.B", "target_statement": "B", "text": "Premise.B : B", "tokens": ["premise", "b"]},
            {"premise_id": "Premise.C", "target_statement": "C", "text": "Premise.C : C", "tokens": ["premise", "c"]},
        ]

        hits = select_hint_premises(
            query="irrelevant",
            task_metadata={},
            corpus=corpus,
            bm25=BM25Index(corpus),
            reranker=None,
            hint_source="random",
            bm25_candidate_count=8,
            rerank_top_n=4,
            hint_top_k=2,
            rng=random.Random(11),
        )

        self.assertEqual(len(hits), 2)
        self.assertTrue(all(row["premise_id"] in {"Premise.A", "Premise.B", "Premise.C"} for row in hits))
        self.assertEqual([row["premise_id"] for row in hits], ["Premise.B", "Premise.C"])

    def test_select_hint_premises_bm25_prefers_lexical_match_without_reranker(self) -> None:
        corpus = [
            {
                "premise_id": "Polynomial.degree_add",
                "target_statement": "Polynomial.degree (p + q) <= _",
                "text": "Polynomial.degree_add : Polynomial.degree (p + q) <= _",
                "tokens": ["polynomial", "degree", "add", "p", "q"],
            },
            {
                "premise_id": "Nat.succ_eq_add_one",
                "target_statement": "Nat.succ n = n + 1",
                "text": "Nat.succ_eq_add_one : Nat.succ n = n + 1",
                "tokens": ["nat", "succ", "add", "one"],
            },
        ]

        hits = select_hint_premises(
            query="prove something about Polynomial degree",
            task_metadata={},
            corpus=corpus,
            bm25=BM25Index(corpus),
            reranker=None,
            hint_source="bm25",
            bm25_candidate_count=8,
            rerank_top_n=4,
            hint_top_k=1,
            rng=random.Random(7),
        )

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["premise_id"], "Polynomial.degree_add")

    def test_build_pairwise_ranking_examples_balances_negatives_per_positive(self) -> None:
        candidate_rows = [
            {
                "sample_id": "sample-1",
                "query": "theorem demo : True := by sorry",
                "premise_text": "Needed.one : True",
                "label": 1,
                "premise_id": "Needed.one",
            },
            {
                "sample_id": "sample-1",
                "query": "theorem demo : True := by sorry",
                "premise_text": "Wrong.one : False",
                "label": 0,
                "premise_id": "Wrong.one",
            },
            {
                "sample_id": "sample-1",
                "query": "theorem demo : True := by sorry",
                "premise_text": "Wrong.two : False",
                "label": 0,
                "premise_id": "Wrong.two",
            },
            {
                "sample_id": "sample-1",
                "query": "theorem demo : True := by sorry",
                "premise_text": "Wrong.three : False",
                "label": 0,
                "premise_id": "Wrong.three",
            },
        ]

        ranking_rows = build_pairwise_ranking_examples(
            candidate_rows,
            negatives_per_positive=2,
            rng=random.Random(7),
        )

        self.assertEqual(len(ranking_rows), 2)
        self.assertTrue(all(row["sample_id"] == "sample-1" for row in ranking_rows))
        self.assertTrue(all(row["positive_premise_id"] == "Needed.one" for row in ranking_rows))
        negative_ids = {row["negative_premise_id"] for row in ranking_rows}
        self.assertEqual(len(negative_ids), 2)
        self.assertTrue(negative_ids.issubset({"Wrong.one", "Wrong.two", "Wrong.three"}))


if __name__ == "__main__":
    unittest.main()
