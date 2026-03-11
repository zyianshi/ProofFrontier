from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


def _resolve_backend_command(args: argparse.Namespace) -> str:
    if args.backend_command:
        return args.backend_command
    env_command = os.environ.get("DEEPSEEK_PROVER_BACKEND_COMMAND", "").strip()
    if env_command:
        return env_command
    if args.mock:
        return ""
    raise RuntimeError(
        "No DeepSeek-Prover backend configured. Set DEEPSEEK_PROVER_BACKEND_COMMAND "
        "or pass --backend-command, or use --mock for local smoke tests."
    )


def _extract_candidate_completion(raw_text: str) -> str:
    blocks = re.findall(r"```(?:lean4|lean)?\s*(.*?)```", raw_text, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return blocks[-1].strip()
    return raw_text.strip()


def _materialize_completed_lean(source_text: str, completion: str) -> str:
    if "theorem " in completion or "import " in completion:
        return completion
    if "sorry" in source_text:
        return source_text.replace("sorry", completion, 1)
    return f"{source_text}\n\n{completion}\n"


def _run_hf_backend(args: argparse.Namespace) -> dict:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "transformers, torch, and bitsandbytes are required for local DeepSeek-Prover inference."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir or None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = None
    if args.torch_dtype and args.torch_dtype != "auto":
        dtype = getattr(torch, args.torch_dtype)

    quantization_config = None
    if args.load_in_4bit:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
        device_map=args.device_map,
        torch_dtype=dtype,
        quantization_config=quantization_config,
        cache_dir=args.cache_dir or None,
    )
    model.eval()

    source_text = Path(args.lean_file).read_text(encoding="utf-8")
    prompt = (
        "Complete the following Lean 4 proof. Return only Lean code for the missing proof term.\n\n"
        f"{source_text}\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0.0,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    completion = _extract_candidate_completion(text)
    completed = _materialize_completed_lean(source_text, completion)
    candidate_file = Path(args.out_json).with_suffix(".candidate.lean")
    candidate_file.write_text(completed, encoding="utf-8")

    validator_cmd = args.validator_command or os.environ.get("TAAM_SOLVER_VALIDATOR_COMMAND", "").strip()
    if not validator_cmd and (args.validator_project_root or args.validator_lean_bin):
        validator_cmd = (
            'python tools/run_lean_validator.py --lean-file "{lean_file}" '
            '--out-json "{out_json}" --project-root "{project_root}" '
            '--lean-bin "{lean_bin}" --disallow-sorry'
        ).format(
            lean_file=str(candidate_file),
            out_json=args.out_json,
            project_root=args.validator_project_root,
            lean_bin=args.validator_lean_bin or "lean",
        )
    elif validator_cmd:
        validator_cmd = validator_cmd.format(
            lean_file=str(candidate_file),
            out_json=args.out_json,
            timeout_sec=args.timeout_sec,
        )

    if validator_cmd:
        proc = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", validator_cmd],
            capture_output=True,
            text=True,
            timeout=args.timeout_sec,
        )
        solved = proc.returncode == 0
        return {
            "solved": solved,
            "engine": args.model_id,
            "candidate_file": str(candidate_file),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
        }

    return {
        "solved": False,
        "engine": args.model_id,
        "candidate_file": str(candidate_file),
        "error": "validator_missing",
        "raw_generation": text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DeepSeek-Prover backend for a generated Lean problem")
    parser.add_argument("--lean-file", type=str, required=True)
    parser.add_argument("--out-json", type=str, required=True)
    parser.add_argument("--backend-command", type=str, default="")
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--model-id", type=str, default="deepseek-ai/DeepSeek-Prover-V2-7B")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--torch-dtype", type=str, default="auto")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--bnb-4bit-compute-dtype", type=str, default="float16")
    parser.add_argument("--bnb-4bit-quant-type", type=str, default="nf4")
    parser.add_argument("--bnb-4bit-use-double-quant", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--cache-dir", type=str, default="")
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument("--validator-command", type=str, default="")
    parser.add_argument("--validator-project-root", type=str, default="")
    parser.add_argument("--validator-lean-bin", type=str, default="")
    args = parser.parse_args()

    if args.mock:
        lean_text = Path(args.lean_file).read_text(encoding="utf-8")
        hyp_count = lean_text.count("(h_")
        solved = hyp_count >= 4
        result = {"solved": solved, "engine": "mock_deepseek_prover"}
        Path(args.out_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"TAAM_PROVER_VERDICT: {'PROVED' if solved else 'FAILED'}")
        return

    if args.backend_command or os.environ.get("DEEPSEEK_PROVER_BACKEND_COMMAND", "").strip():
        command_template = _resolve_backend_command(args)
        command = command_template.format(
            lean_file=args.lean_file,
            out_json=args.out_json,
            timeout_sec=args.timeout_sec,
        )
        try:
            proc = subprocess.run(
                ["powershell.exe", "-NoProfile", "-Command", command],
                capture_output=True,
                text=True,
                timeout=args.timeout_sec,
            )
            solved = proc.returncode == 0
            result = {
                "solved": solved,
                "engine": "external_deepseek_prover",
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "returncode": proc.returncode,
            }
        except subprocess.TimeoutExpired:
            solved = False
            result = {"solved": False, "engine": "external_deepseek_prover", "error": "timeout"}
    else:
        result = _run_hf_backend(args)
        solved = bool(result.get("solved", False))

    Path(args.out_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TAAM_PROVER_VERDICT: {'PROVED' if solved else 'FAILED'}")


if __name__ == "__main__":
    main()
