from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .repair_helper import is_algebra_like_task_name


def _payload_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    tasks = int(payload.get("tasks", 0))
    successes = int(payload.get("successes", 0))
    return {
        "tasks": tasks,
        "successes": successes,
        "pass_k": int(payload.get("pass_k", 0)),
        "pass_rate": (successes / tasks) if tasks else 0.0,
        "hint_source": payload.get("hint_source", ""),
    }


def _summarize_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    row_list = list(rows)
    tasks = len(row_list)
    successes = sum(1 for row in row_list if bool(row.get("success")))
    return {
        "tasks": tasks,
        "successes": successes,
        "pass_rate": (successes / tasks) if tasks else 0.0,
    }


def _algebra_like_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = payload.get("results", [])
    return [
        row
        for row in rows
        if is_algebra_like_task_name(str(row.get("theorem_name", row.get("task_id", ""))))
    ]


def summarize_effectiveness_suite(
    *,
    hard_replay_vanilla: Dict[str, Any],
    hard_replay_oracle: Dict[str, Any],
    hard_replay_learned: Dict[str, Any],
    minif2f_vanilla: Dict[str, Any],
    minif2f_learned: Dict[str, Any],
) -> Dict[str, Any]:
    hard_vanilla_summary = _payload_summary(hard_replay_vanilla)
    hard_oracle_summary = _payload_summary(hard_replay_oracle)
    hard_learned_summary = _payload_summary(hard_replay_learned)
    mini_vanilla_summary = _payload_summary(minif2f_vanilla)
    mini_learned_summary = _payload_summary(minif2f_learned)
    algebra_vanilla_summary = _summarize_rows(_algebra_like_rows(minif2f_vanilla))
    algebra_learned_summary = _summarize_rows(_algebra_like_rows(minif2f_learned))

    hard_oracle_delta = hard_oracle_summary["pass_rate"] - hard_vanilla_summary["pass_rate"]
    hard_learned_delta = hard_learned_summary["pass_rate"] - hard_vanilla_summary["pass_rate"]
    algebra_learned_delta = algebra_learned_summary["pass_rate"] - algebra_vanilla_summary["pass_rate"]
    full_minif2f_delta = mini_learned_summary["pass_rate"] - mini_vanilla_summary["pass_rate"]

    criteria = {
        "targeted_hard_replay_oracle_uplift": hard_oracle_delta > 0.0,
        "targeted_hard_replay_learned_uplift": hard_learned_delta > 0.0,
        "targeted_minif2f_algebra_like_learned_uplift": algebra_learned_delta > 0.0,
        "general_minif2f_no_regression": full_minif2f_delta >= 0.0,
    }

    return {
        "targeted_effectiveness": {
            "hard_sample_replay_test": {
                "vanilla": hard_vanilla_summary,
                "oracle": hard_oracle_summary,
                "learned": hard_learned_summary,
                "oracle_delta_vs_vanilla": hard_oracle_delta,
                "learned_delta_vs_vanilla": hard_learned_delta,
            },
            "miniF2F_algebra_like": {
                "vanilla": algebra_vanilla_summary,
                "learned": algebra_learned_summary,
                "learned_delta_vs_vanilla": algebra_learned_delta,
            },
        },
        "general_no_regression": {
            "miniF2F_full": {
                "vanilla": mini_vanilla_summary,
                "learned": mini_learned_summary,
                "learned_delta_vs_vanilla": full_minif2f_delta,
            }
        },
        "criteria": criteria,
        "hard_samples_effective": all(criteria.values()),
    }
