from __future__ import annotations

import unittest

from taam.downstream.effectiveness_suite import summarize_effectiveness_suite


def _payload(theorem_names: list[str], success_flags: list[bool], *, hint_source: str) -> dict:
    rows = []
    for idx, theorem_name in enumerate(theorem_names):
        rows.append(
            {
                "task_id": f"task::{idx}",
                "theorem_name": theorem_name,
                "success": bool(success_flags[idx]),
            }
        )
    successes = sum(1 for flag in success_flags if flag)
    return {
        "tasks": len(theorem_names),
        "successes": successes,
        "pass_k": 32,
        "hint_source": hint_source,
        "results": rows,
    }


class EffectivenessSuiteTests(unittest.TestCase):
    def test_summarize_effectiveness_suite_requires_targeted_uplift_and_no_regression(self) -> None:
        hard_vanilla = _payload(["hard_a", "hard_b", "hard_c", "hard_d"], [True, False, False, False], hint_source="vanilla")
        hard_oracle = _payload(["hard_a", "hard_b", "hard_c", "hard_d"], [True, True, False, False], hint_source="oracle")
        hard_learned = _payload(["hard_a", "hard_b", "hard_c", "hard_d"], [True, True, False, False], hint_source="learned")
        mini_vanilla = _payload(
            ["mathd_algebra_1", "mathd_numbertheory_1", "polynomial_case", "geometry_case"],
            [False, True, False, False],
            hint_source="vanilla",
        )
        mini_learned = _payload(
            ["mathd_algebra_1", "mathd_numbertheory_1", "polynomial_case", "geometry_case"],
            [True, False, False, False],
            hint_source="learned",
        )

        summary = summarize_effectiveness_suite(
            hard_replay_vanilla=hard_vanilla,
            hard_replay_oracle=hard_oracle,
            hard_replay_learned=hard_learned,
            minif2f_vanilla=mini_vanilla,
            minif2f_learned=mini_learned,
        )

        self.assertTrue(summary["criteria"]["targeted_hard_replay_oracle_uplift"])
        self.assertTrue(summary["criteria"]["targeted_hard_replay_learned_uplift"])
        self.assertTrue(summary["criteria"]["targeted_minif2f_algebra_like_learned_uplift"])
        self.assertTrue(summary["criteria"]["general_minif2f_no_regression"])
        self.assertTrue(summary["hard_samples_effective"])

    def test_summarize_effectiveness_suite_rejects_when_full_minif2f_regresses(self) -> None:
        hard_vanilla = _payload(["hard_a", "hard_b", "hard_c", "hard_d"], [True, False, False, False], hint_source="vanilla")
        hard_oracle = _payload(["hard_a", "hard_b", "hard_c", "hard_d"], [True, True, False, False], hint_source="oracle")
        hard_learned = _payload(["hard_a", "hard_b", "hard_c", "hard_d"], [True, True, False, False], hint_source="learned")
        mini_vanilla = _payload(
            ["mathd_algebra_1", "mathd_numbertheory_1", "polynomial_case", "geometry_case"],
            [True, True, False, False],
            hint_source="vanilla",
        )
        mini_learned = _payload(
            ["mathd_algebra_1", "mathd_numbertheory_1", "polynomial_case", "geometry_case"],
            [True, False, False, False],
            hint_source="learned",
        )

        summary = summarize_effectiveness_suite(
            hard_replay_vanilla=hard_vanilla,
            hard_replay_oracle=hard_oracle,
            hard_replay_learned=hard_learned,
            minif2f_vanilla=mini_vanilla,
            minif2f_learned=mini_learned,
        )

        self.assertFalse(summary["criteria"]["general_minif2f_no_regression"])
        self.assertFalse(summary["hard_samples_effective"])


if __name__ == "__main__":
    unittest.main()
