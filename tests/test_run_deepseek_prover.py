from __future__ import annotations

import unittest

import torch
from transformers.tokenization_utils_base import BatchEncoding

import tools.run_deepseek_prover as run_deepseek_prover


class RunDeepSeekProverTests(unittest.TestCase):
    def test_prepare_generation_inputs_handles_batch_encoding(self) -> None:
        input_ids = torch.tensor([[1, 2, 3, 4]])
        rendered_inputs = BatchEncoding(
            {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
        )

        prompt_length, generate_args = run_deepseek_prover._prepare_generation_inputs(
            rendered_inputs,
            torch.device("cpu"),
        )

        self.assertEqual(prompt_length, 4)
        self.assertIn("input_ids", generate_args)
        self.assertIn("attention_mask", generate_args)
        self.assertEqual(generate_args["input_ids"].shape, torch.Size([1, 4]))


if __name__ == "__main__":
    unittest.main()
