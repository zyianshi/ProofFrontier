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
    return os.environ.get('DEEPSEEK_PROVER_BACKEND_COMMAND', '').strip()


def _extract_candidate_completion(raw_text: str) -> str:
    blocks = re.findall(r"```(?:lean4|lean)?\s*(.*?)```", raw_text, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return blocks[-1].strip()
    return raw_text.strip()


def _materialize_completed_lean(source_text: str, completion: str) -> str:
    if 'theorem ' in completion or 'import ' in completion:
        return completion
    if 'sorry' in source_text:
        return source_text.replace('sorry', completion, 1)
    return f"{source_text}\n\n{completion}\n"


def _build_prompt(source_text: str) -> str:
    return (
        'You are DeepSeek-Prover. Complete the missing Lean 4 proof. '
        'Return only the Lean code that should replace the single sorry.\n\n'
        f'{source_text}\n'
    )


def _run_shell(command: str, timeout_sec: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )


def _build_validator_command(
    args: argparse.Namespace,
    candidate_file: Path,
    validator_result_json: Path,
) -> str:
    if args.validator_command:
        return args.validator_command.format(
            lean_file=str(candidate_file),
            out_json=str(validator_result_json),
            timeout_sec=args.timeout_sec,
        )
    env_command = os.environ.get('TAAM_SOLVER_VALIDATOR_COMMAND', '').strip()
    if env_command:
        return env_command.format(
            lean_file=str(candidate_file),
            out_json=str(validator_result_json),
            timeout_sec=args.timeout_sec,
        )
    if args.validator_project_root or args.validator_lean_bin or args.validator_lake_bin or args.validator_use_wsl:
        parts = [
            'python tools/run_lean_validator.py',
            f'--lean-file "{candidate_file}"',
            f'--out-json "{validator_result_json}"',
        ]
        if args.validator_project_root:
            parts.append(f'--project-root "{args.validator_project_root}"')
        if args.validator_lean_bin:
            parts.append(f'--lean-bin "{args.validator_lean_bin}"')
        if args.validator_lake_bin:
            parts.append(f'--lake-bin "{args.validator_lake_bin}"')
        if args.validator_use_wsl:
            parts.append('--use-wsl')
        if args.validator_wsl_user:
            parts.append(f'--wsl-user "{args.validator_wsl_user}"')
        if args.validator_wsl_distro:
            parts.append(f'--wsl-distro "{args.validator_wsl_distro}"')
        parts.append('--disallow-sorry')
        return ' '.join(parts)
    return ''


def _model_input_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        import torch

        return torch.device('cpu')


def _run_hf_backend(args: argparse.Namespace) -> dict:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            'transformers, torch, and bitsandbytes are required for local DeepSeek-Prover inference.'
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir or None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = None
    if args.torch_dtype and args.torch_dtype != 'auto':
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

    source_text = Path(args.lean_file).read_text(encoding='utf-8')
    prompt = _build_prompt(source_text)
    if hasattr(tokenizer, 'apply_chat_template'):
        prompt = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = tokenizer(prompt, return_tensors='pt')
    input_device = _model_input_device(model)
    inputs = {k: v.to(input_device) for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0.0,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    raw_generation = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    completion = _extract_candidate_completion(raw_generation)
    completed = _materialize_completed_lean(source_text, completion)

    candidate_file = Path(args.out_json).with_suffix('.candidate.lean')
    candidate_file.write_text(completed, encoding='utf-8')

    validator_result_json = Path(args.out_json).with_suffix('.validator.json')
    validator_cmd = _build_validator_command(args, candidate_file, validator_result_json)
    if not validator_cmd:
        return {
            'solved': False,
            'engine': args.model_id,
            'candidate_file': str(candidate_file),
            'completion': completion,
            'raw_generation': raw_generation,
            'error': 'validator_missing',
        }

    try:
        proc = _run_shell(validator_cmd, args.timeout_sec)
    except subprocess.TimeoutExpired:
        return {
            'solved': False,
            'engine': args.model_id,
            'candidate_file': str(candidate_file),
            'completion': completion,
            'raw_generation': raw_generation,
            'validator_command': validator_cmd,
            'error': 'validator_timeout',
        }

    validator_result = None
    if validator_result_json.exists():
        validator_result = json.loads(validator_result_json.read_text(encoding='utf-8'))
    solved = bool((validator_result or {}).get('passed', proc.returncode == 0))
    return {
        'solved': solved,
        'engine': args.model_id,
        'candidate_file': str(candidate_file),
        'completion': completion,
        'raw_generation': raw_generation,
        'validator_command': validator_cmd,
        'validator_result': validator_result,
        'stdout': proc.stdout,
        'stderr': proc.stderr,
        'returncode': proc.returncode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Run DeepSeek-Prover backend for a generated Lean problem')
    parser.add_argument('--lean-file', type=str, required=True)
    parser.add_argument('--out-json', type=str, required=True)
    parser.add_argument('--backend-command', type=str, default='')
    parser.add_argument('--timeout-sec', type=int, default=180)
    parser.add_argument('--mock', action='store_true')
    parser.add_argument('--model-id', type=str, default='deepseek-ai/DeepSeek-Prover-V2-7B')
    parser.add_argument('--device-map', type=str, default='auto')
    parser.add_argument('--torch-dtype', type=str, default='auto')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--bnb-4bit-compute-dtype', type=str, default='float16')
    parser.add_argument('--bnb-4bit-quant-type', type=str, default='nf4')
    parser.add_argument('--bnb-4bit-use-double-quant', action='store_true')
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--cache-dir', type=str, default='')
    parser.add_argument('--trust-remote-code', action='store_true', default=True)
    parser.add_argument('--validator-command', type=str, default='')
    parser.add_argument('--validator-project-root', type=str, default='')
    parser.add_argument('--validator-lean-bin', type=str, default='')
    parser.add_argument('--validator-lake-bin', type=str, default='')
    parser.add_argument('--validator-use-wsl', action='store_true')
    parser.add_argument('--validator-wsl-user', type=str, default='')
    parser.add_argument('--validator-wsl-distro', type=str, default='Ubuntu')
    args = parser.parse_args()

    if args.mock:
        lean_text = Path(args.lean_file).read_text(encoding='utf-8')
        hyp_count = lean_text.count('(h_')
        solved = hyp_count >= 4
        result = {'solved': solved, 'engine': 'mock_deepseek_prover'}
        Path(args.out_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"TAAM_PROVER_VERDICT: {'PROVED' if solved else 'FAILED'}")
        return

    backend_template = _resolve_backend_command(args)
    if backend_template:
        command = backend_template.format(
            lean_file=args.lean_file,
            out_json=args.out_json,
            timeout_sec=args.timeout_sec,
        )
        try:
            proc = _run_shell(command, args.timeout_sec)
            solved = proc.returncode == 0
            result = {
                'solved': solved,
                'engine': 'external_deepseek_prover',
                'stdout': proc.stdout,
                'stderr': proc.stderr,
                'returncode': proc.returncode,
            }
        except subprocess.TimeoutExpired:
            solved = False
            result = {'solved': False, 'engine': 'external_deepseek_prover', 'error': 'timeout'}
    else:
        result = _run_hf_backend(args)
        solved = bool(result.get('solved', False))

    Path(args.out_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"TAAM_PROVER_VERDICT: {'PROVED' if solved else 'FAILED'}")


if __name__ == '__main__':
    main()
