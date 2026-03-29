from __future__ import annotations

import unittest

from taam.prover_utils import (
    build_prover_prompt,
    extract_candidate_completion,
    render_prompt_for_model,
)


class _FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def apply_chat_template(
        self,
        chat,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_tensors: str,
    ):
        self.calls.append(
            {
                "chat": chat,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "return_tensors": return_tensors,
            }
        )
        return "TOKENIZED"


class ProverUtilsTests(unittest.TestCase):
    def test_build_prover_prompt_matches_official_v2_shape(self) -> None:
        source_text = "import Mathlib\n\ntheorem demo : True := by\n  sorry\n"

        prompt = build_prover_prompt(source_text)

        self.assertIn("Complete the following Lean 4 code:", prompt)
        self.assertIn("```lean4\nimport Mathlib\n\ntheorem demo : True := by\n  sorry\n```", prompt)
        self.assertIn("provide a detailed proof plan", prompt)
        self.assertIn("construction of the final formal proof", prompt)
        self.assertIn("exactly one final ```lean4``` code block", prompt)
        self.assertIn("Do not restate the theorem header or rename hypotheses", prompt)
        self.assertIn("Do not include sorry, admit, or placeholders", prompt)

    def test_render_prompt_for_model_uses_chat_template(self) -> None:
        tokenizer = _FakeTokenizer()
        source_text = "import Mathlib\n\ntheorem demo : True := by\n  sorry\n"

        rendered = render_prompt_for_model(tokenizer, source_text)

        self.assertEqual(rendered, "TOKENIZED")
        self.assertEqual(len(tokenizer.calls), 1)
        call = tokenizer.calls[0]
        self.assertEqual(call["tokenize"], True)
        self.assertEqual(call["add_generation_prompt"], True)
        self.assertEqual(call["return_tensors"], "pt")
        self.assertEqual(
            call["chat"],
            [{"role": "user", "content": build_prover_prompt(source_text)}],
        )

    def test_extract_candidate_completion_prefers_last_fenced_lean_block(self) -> None:
        raw_text = (
            "Here is the proof plan:\n"
            "1. simplify the goal\n"
            "2. finish with reflexivity\n\n"
            "```lean4\n"
            "theorem demo : True := by\n"
            "  trivial\n"
            "```\n"
            "Some extra explanation.\n"
            "```lean4\n"
            "by\n"
            "  simpa\n"
            "```"
        )

        completion = extract_candidate_completion(raw_text)

        self.assertEqual(completion, "by\n  simpa")


if __name__ == "__main__":
    unittest.main()
