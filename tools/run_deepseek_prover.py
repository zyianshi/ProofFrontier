from __future__ import annotations

import argparse
import ctypes
import importlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from collections.abc import Mapping
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taam.external_tools import materialize_lean_completion
from taam.prover_utils import (
    build_prover_prompt,
    extract_candidate_completion,
    extract_declaration_echo_body,
    is_full_lean_document,
    normalize_generation_text,
    normalize_proof_completion_body,
    render_prompt_for_model,
)


def _resolve_backend_command(args: argparse.Namespace) -> str:
    if args.backend_command:
        return args.backend_command
    return os.environ.get("DEEPSEEK_PROVER_BACKEND_COMMAND", "").strip()


def _materialize_completed_lean(source_text: str, completion: str) -> str:
    echoed_body = extract_declaration_echo_body(completion)
    if echoed_body:
        return materialize_lean_completion(source_text, f"by\n{echoed_body}")
    if is_full_lean_document(completion):
        return materialize_lean_completion(source_text, completion)
    normalized = normalize_proof_completion_body(completion)
    if not normalized:
        return materialize_lean_completion(source_text, "")
    return materialize_lean_completion(source_text, f"by\n{normalized}")


def _build_prompt(source_text: str) -> str:
    return build_prover_prompt(source_text)


def _run_shell(command: str, timeout_sec: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )


def _urlopen_no_proxy(req_or_url, timeout_sec: int):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(req_or_url, timeout=timeout_sec)


def _ijit_stub_source() -> str:
    return """
#ifdef __cplusplus
extern "C" {
#endif
static unsigned int taam_method_id = 1;
unsigned int iJIT_GetNewMethodID(void) { return taam_method_id++; }
int iJIT_IsProfilingActive(void) { return 0; }
int iJIT_NotifyEvent(int event_type, void *event_data) {
  (void)event_type;
  (void)event_data;
  return 0;
}
#ifdef __cplusplus
}
#endif
""".strip()


def _try_load_ijit_stub(import_error: ImportError) -> bool:
    if os.name == "nt" or "iJIT_" not in str(import_error):
        return False

    runtime_dir = Path(tempfile.gettempdir()) / "taam_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    c_path = runtime_dir / "ijit_stub.c"
    so_path = runtime_dir / "libijit_stub.so"
    if not so_path.exists():
        c_path.write_text(_ijit_stub_source() + "\n", encoding="utf-8")
        try:
            subprocess.run(
                ["gcc", "-shared", "-fPIC", str(c_path), "-o", str(so_path)],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return False

    try:
        ctypes.CDLL(str(so_path), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
    except OSError:
        return False
    existing = os.environ.get("LD_PRELOAD", "").strip()
    preload_parts = [str(so_path)]
    if existing:
        preload_parts.append(existing)
    os.environ["LD_PRELOAD"] = ":".join(preload_parts)
    return True


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        if not _try_load_ijit_stub(exc):
            raise RuntimeError(
                "transformers, torch, and bitsandbytes are required for local DeepSeek-Prover inference."
            ) from exc
        for name in list(sys.modules):
            if name == "torch" or name.startswith("torch."):
                sys.modules.pop(name, None)
        torch = importlib.import_module("torch")
    return torch


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
    env_command = os.environ.get("TAAM_SOLVER_VALIDATOR_COMMAND", "").strip()
    if env_command:
        return env_command.format(
            lean_file=str(candidate_file),
            out_json=str(validator_result_json),
            timeout_sec=args.timeout_sec,
        )
    if args.validator_project_root or args.validator_lean_bin or args.validator_lake_bin or args.validator_use_wsl:
        parts = [
            "python tools/run_lean_validator.py",
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
            parts.append("--use-wsl")
            if args.validator_wsl_user:
                parts.append(f'--wsl-user "{args.validator_wsl_user}"')
            if args.validator_wsl_distro:
                parts.append(f'--wsl-distro "{args.validator_wsl_distro}"')
        parts.append("--disallow-sorry")
        return " ".join(parts)
    return ""


def _model_input_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        torch = _import_torch()
        return torch.device("cpu")


def _load_hf_stack():
    try:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "transformers, torch, and bitsandbytes are required for local DeepSeek-Prover inference."
        ) from exc
    return AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _normalize_rope_parameters(config: Any) -> Any:
    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, dict):
        normalized = dict(rope_parameters)
        for key in ("factor", "beta_fast", "beta_slow"):
            if key in normalized and normalized[key] is not None:
                normalized[key] = float(normalized[key])
        config.rope_parameters = normalized
    return config


def _resolve_model_source(model_id: str) -> tuple[str, Path | None]:
    model_path = Path(model_id).expanduser()
    adapter_config_path = model_path / "adapter_config.json"
    if model_path.exists() and adapter_config_path.exists():
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        base_model_name = str(adapter_config.get("base_model_name_or_path", "")).strip()
        if not base_model_name:
            raise RuntimeError(f"Adapter directory is missing base_model_name_or_path: {adapter_config_path}")
        return base_model_name, model_path.resolve()
    return model_id, None


def _materialize_local_source(model_ref: str, cache_dir: str) -> str:
    model_path = Path(model_ref).expanduser()
    if model_path.exists():
        return str(model_path.resolve())
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=model_ref,
        cache_dir=cache_dir or None,
        local_files_only=True,
    )


def _quantization_config(args: argparse.Namespace, torch_module) -> Any:
    if not args.load_in_4bit:
        return None
    _AutoConfig, _AutoModelForCausalLM, _AutoTokenizer, BitsAndBytesConfig = _load_hf_stack()
    compute_dtype = getattr(torch_module, args.bnb_4bit_compute_dtype)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
    )


def _load_model_and_tokenizer(
    args: argparse.Namespace,
    *,
    force_single_gpu: bool = False,
):
    torch = _import_torch()
    AutoConfig, AutoModelForCausalLM, AutoTokenizer, _BitsAndBytesConfig = _load_hf_stack()
    base_model_ref, adapter_dir = _resolve_model_source(args.model_id)
    base_model_source = _materialize_local_source(base_model_ref, args.cache_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_source,
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.cache_dir or None,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = None
    if args.torch_dtype and args.torch_dtype != "auto":
        dtype = getattr(torch, args.torch_dtype)

    quantization_config = _quantization_config(args, torch)
    model_config = _normalize_rope_parameters(
        AutoConfig.from_pretrained(
            base_model_source,
            trust_remote_code=args.trust_remote_code,
            cache_dir=args.cache_dir or None,
            local_files_only=True,
        )
    )
    if force_single_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for resident prover service.")
        torch.cuda.set_device(args.cuda_device)
        device_map: Any = {"": f"cuda:{args.cuda_device}"}
    else:
        device_map = args.device_map

    model = AutoModelForCausalLM.from_pretrained(
        base_model_source,
        trust_remote_code=args.trust_remote_code,
        device_map=device_map,
        torch_dtype=dtype,
        config=model_config,
        quantization_config=quantization_config,
        cache_dir=args.cache_dir or None,
        local_files_only=True,
    )
    if adapter_dir is not None:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("peft is required to load a local LoRA adapter directory.") from exc
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    model.eval()
    return model, tokenizer


def _mock_generation(source_text: str, engine: str) -> dict[str, Any]:
    completion = "simpa"
    if "by\n  sorry" not in source_text and "sorry" not in source_text:
        completion = source_text.strip()
    return {
        "completion": completion,
        "raw_generation": completion,
        "engine": engine,
    }


def _prepare_generation_inputs(rendered_inputs: Any, input_device: Any) -> tuple[int, dict[str, Any]]:
    if hasattr(rendered_inputs, "to"):
        rendered_inputs = rendered_inputs.to(input_device)
    if isinstance(rendered_inputs, Mapping) or hasattr(rendered_inputs, "input_ids"):
        inputs = {k: v.to(input_device) if hasattr(v, "to") else v for k, v in rendered_inputs.items()}
        prompt_length = inputs["input_ids"].shape[1]
        return prompt_length, dict(inputs)
    prompt_length = rendered_inputs.shape[1]
    return prompt_length, {"input_ids": rendered_inputs}


def _generate_completion(
    source_text: str,
    args: argparse.Namespace,
    model,
    tokenizer,
    *,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    if args.mock:
        return _mock_generation(source_text, engine="mock_resident_deepseek_prover")

    input_device = _model_input_device(model)
    rendered_inputs = render_prompt_for_model(tokenizer, source_text)
    prompt_length, generate_args = _prepare_generation_inputs(rendered_inputs, input_device)
    actual_max_new_tokens = int(max_new_tokens if max_new_tokens is not None else args.max_new_tokens)
    actual_temperature = float(temperature if temperature is not None else args.temperature)
    actual_top_p = float(top_p if top_p is not None else args.top_p)
    actual_seed = int(seed if seed is not None else args.seed)
    if actual_seed >= 0:
        torch = _import_torch()
        torch.manual_seed(actual_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(actual_seed)
    outputs = model.generate(
        **generate_args,
        max_new_tokens=actual_max_new_tokens,
        do_sample=actual_temperature > 0.0,
        temperature=actual_temperature,
        top_p=actual_top_p,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = outputs[0][prompt_length:]
    raw_generation = normalize_generation_text(
        tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    ).strip()
    completion = extract_candidate_completion(raw_generation)
    return {
        "completion": completion,
        "raw_generation": raw_generation,
        "engine": args.model_id,
    }


def _validate_candidate(
    args: argparse.Namespace,
    source_text: str,
    generation: dict[str, Any],
) -> dict[str, Any]:
    completion = generation["completion"]
    raw_generation = generation["raw_generation"]
    completed = _materialize_completed_lean(source_text, completion)

    candidate_file = Path(args.out_json).with_suffix(".candidate.lean")
    candidate_file.write_text(completed, encoding="utf-8")

    validator_result_json = Path(args.out_json).with_suffix(".validator.json")
    validator_cmd = _build_validator_command(args, candidate_file, validator_result_json)
    if not validator_cmd:
        return {
            "solved": False,
            "engine": generation.get("engine", args.model_id),
            "candidate_file": str(candidate_file),
            "completion": completion,
            "raw_generation": raw_generation,
            "error": "validator_missing",
        }

    try:
        proc = _run_shell(validator_cmd, args.timeout_sec)
    except subprocess.TimeoutExpired:
        return {
            "solved": False,
            "engine": generation.get("engine", args.model_id),
            "candidate_file": str(candidate_file),
            "completion": completion,
            "raw_generation": raw_generation,
            "validator_command": validator_cmd,
            "error": "validator_timeout",
        }

    validator_result = None
    if validator_result_json.exists():
        validator_result = json.loads(validator_result_json.read_text(encoding="utf-8"))
    solved = bool((validator_result or {}).get("passed", proc.returncode == 0))
    return {
        "solved": solved,
        "engine": generation.get("engine", args.model_id),
        "candidate_file": str(candidate_file),
        "completion": completion,
        "raw_generation": raw_generation,
        "validator_command": validator_cmd,
        "validator_result": validator_result,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "returncode": proc.returncode,
    }


def _run_hf_backend(args: argparse.Namespace) -> dict[str, Any]:
    if args.mock:
        source_text = Path(args.lean_file).read_text(encoding="utf-8")
        generation = _mock_generation(source_text, engine="mock_deepseek_prover")
        return _validate_candidate(args, source_text, generation)
    model, tokenizer = _load_model_and_tokenizer(args, force_single_gpu=False)
    source_text = Path(args.lean_file).read_text(encoding="utf-8")
    generation = _generate_completion(source_text, args, model, tokenizer)
    return _validate_candidate(args, source_text, generation)


def _post_json(url: str, payload: dict[str, Any], timeout_sec: int) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with _urlopen_no_proxy(req, timeout_sec) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def _run_server_backend(args: argparse.Namespace) -> dict[str, Any]:
    if not args.server_url:
        raise RuntimeError("server_url is required for resident prover client mode.")
    source_text = Path(args.lean_file).read_text(encoding="utf-8")
    try:
        generation = _post_json(
            args.server_url.rstrip("/") + "/generate",
            {
                "lean_text": source_text,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "seed": args.seed,
            },
            timeout_sec=args.timeout_sec,
        )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {
            "solved": False,
            "engine": "resident_deepseek_prover",
            "error": f"http_{exc.code}",
            "stderr": body,
        }
    except urllib.error.URLError as exc:
        return {
            "solved": False,
            "engine": "resident_deepseek_prover",
            "error": f"server_unreachable:{exc}",
        }
    return _validate_candidate(args, source_text, generation)


class _ResidentProverState:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model = None
        self.tokenizer = None
        self.loaded = False
        self.load_error = ""
        self.request_count = 0
        self.active_request = False
        self.state_lock = threading.Lock()
        self.request_lock = threading.Lock()

    def start_loading(self) -> None:
        if self.args.mock:
            with self.state_lock:
                self.loaded = True
                self.load_error = ""
            return
        thread = threading.Thread(target=self._load_model, name="deepseek-prover-loader", daemon=True)
        thread.start()

    def _load_model(self) -> None:
        try:
            model, tokenizer = _load_model_and_tokenizer(self.args, force_single_gpu=True)
            with self.state_lock:
                self.model = model
                self.tokenizer = tokenizer
                self.loaded = True
                self.load_error = ""
        except Exception as exc:  # pragma: no cover
            with self.state_lock:
                self.loaded = False
                self.load_error = f"{type(exc).__name__}:{exc}"

    def health(self) -> dict[str, Any]:
        with self.state_lock:
            return {
                "loaded": self.loaded,
                "load_error": self.load_error,
                "model_id": self.args.model_id,
                "gpu_device": self.args.cuda_device,
                "pid": os.getpid(),
                "request_count": self.request_count,
                "active_request": self.active_request,
                "engine": "resident_deepseek_prover_service",
            }

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        lean_text = str(payload.get("lean_text", ""))
        if not lean_text:
            raise ValueError("lean_text is required")
        with self.state_lock:
            if self.load_error:
                raise RuntimeError(self.load_error)
            if not self.loaded or self.model is None or self.tokenizer is None:
                raise RuntimeError("model_not_ready")
        with self.request_lock:
            with self.state_lock:
                self.active_request = True
                self.request_count += 1
                model = self.model
                tokenizer = self.tokenizer
            try:
                if self.args.mock:
                    return _mock_generation(lean_text, engine="mock_resident_deepseek_prover")
                return _generate_completion(
                    lean_text,
                    self.args,
                    model,
                    tokenizer,
                    max_new_tokens=payload.get("max_new_tokens"),
                    temperature=payload.get("temperature"),
                    top_p=payload.get("top_p"),
                    seed=payload.get("seed"),
                )
            finally:
                with self.state_lock:
                    self.active_request = False


class _ResidentHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], handler_cls, state: _ResidentProverState):
        super().__init__(server_address, handler_cls)
        self.state = state


class _ResidentHandler(BaseHTTPRequestHandler):
    server: _ResidentHTTPServer

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args) -> None:
        sys.stdout.write(f"[resident-prover] {self.address_string()} - {fmt % args}\n")
        sys.stdout.flush()

    def do_GET(self) -> None:
        if self.path.rstrip("/") != "/health":
            self._write_json(404, {"error": "not_found"})
            return
        self._write_json(200, self.server.state.health())

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/generate":
            self._write_json(404, {"error": "not_found"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length).decode("utf-8")
            payload = json.loads(raw_body or "{}")
        except (ValueError, json.JSONDecodeError) as exc:
            self._write_json(400, {"error": f"bad_request:{exc}"})
            return

        health = self.server.state.health()
        if health["load_error"]:
            self._write_json(500, {"error": health["load_error"]})
            return
        if not health["loaded"]:
            self._write_json(503, {"error": "model_not_ready"})
            return

        try:
            result = self.server.state.generate(payload)
        except ValueError as exc:
            self._write_json(400, {"error": str(exc)})
            return
        except RuntimeError as exc:
            self._write_json(503, {"error": str(exc)})
            return
        self._write_json(200, result)


def _run_service(args: argparse.Namespace) -> None:
    state = _ResidentProverState(args)
    state.start_loading()
    server = _ResidentHTTPServer((args.host, args.port), _ResidentHandler, state)
    print(
        f"Resident DeepSeek prover listening on http://{args.host}:{args.port} "
        f"(gpu_device={args.cuda_device}, model_id={args.model_id})",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DeepSeek-Prover backend for a generated Lean problem")
    parser.add_argument("--lean-file", type=str, default="")
    parser.add_argument("--out-json", type=str, default="")
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
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cache-dir", type=str, default="")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--validator-command", type=str, default="")
    parser.add_argument("--validator-project-root", type=str, default="")
    parser.add_argument("--validator-lean-bin", type=str, default="")
    parser.add_argument("--validator-lake-bin", type=str, default="")
    parser.add_argument("--validator-use-wsl", action="store_true")
    parser.add_argument("--validator-wsl-user", type=str, default="")
    parser.add_argument("--validator-wsl-distro", type=str, default="Ubuntu")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--server-url", type=str, default="")
    parser.add_argument("--cuda-device", type=int, default=0)
    args = parser.parse_args()

    if args.serve:
        _run_service(args)
        return

    if not args.lean_file or not args.out_json:
        raise SystemExit("--lean-file and --out-json are required unless --serve is set.")

    if args.mock:
        lean_text = Path(args.lean_file).read_text(encoding="utf-8")
        hyp_count = lean_text.count("(h_")
        solved = hyp_count >= 4
        result = {"solved": solved, "engine": "mock_deepseek_prover"}
        Path(args.out_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"TAAM_PROVER_VERDICT: {'PROVED' if solved else 'FAILED'}")
        return

    if args.server_url:
        result = _run_server_backend(args)
        solved = bool(result.get("solved", False))
    else:
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
