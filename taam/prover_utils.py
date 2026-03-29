from __future__ import annotations

import re
import textwrap
from typing import Any

PROVER_PROMPT_HEADER = "Complete the following Lean 4 code:"
PROVER_PLAN_INSTRUCTION = (
    "Before producing the Lean 4 code to formally prove the given theorem, "
    "provide a detailed proof plan outlining the main proof steps and strategies.\n"
    "The plan should highlight key ideas, intermediate lemmas, and proof structures "
    "that will guide the construction of the final formal proof.\n"
    "Then output exactly one final ```lean4``` code block containing the completed Lean 4 theorem, "
    "and do not output any extra prose after that final code block.\n"
    "Do not restate the theorem header or rename hypotheses.\n"
    "Do not include sorry, admit, or placeholders in the final code block."
)


def normalize_generation_text(raw_text: str) -> str:
    text = str(raw_text)
    return text.replace("Ċ", "\n").replace("Ġ", " ")


def build_prover_prompt(source_text: str) -> str:
    source = str(source_text).rstrip()
    return f"{PROVER_PROMPT_HEADER}\n\n```lean4\n{source}\n```\n\n{PROVER_PLAN_INSTRUCTION}"


def render_prompt_for_model(tokenizer, source_text: str) -> str:
    prompt = build_prover_prompt(source_text)
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError("Tokenizer does not expose apply_chat_template required by DeepSeek-Prover-V2.")
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )


def build_prover_target(proof_text: str) -> str:
    body = normalize_proof_completion_body(proof_text)
    if not body:
        return "```"
    return f"{body}\n```"


def strip_prompt_echo(raw_text: str) -> str:
    text = normalize_generation_text(raw_text).strip()
    prompt_echo = re.search(
        r"(?im)\n+\s*(?:#{1,6}\s*)?Complete the following Lean\s*4\s*code\s*:",
        text,
    )
    if prompt_echo:
        prefix = text[: prompt_echo.start()].rstrip()
        if prefix:
            return prefix
    return text


def extract_candidate_completion(raw_text: str) -> str:
    normalized = strip_prompt_echo(raw_text)
    blocks = [
        block.strip()
        for block in re.findall(r"```(?:lean4|lean)?\s*(.*?)```", normalized, flags=re.DOTALL | re.IGNORECASE)
    ]
    non_empty_blocks = [block for block in blocks if block]
    if non_empty_blocks:
        return non_empty_blocks[-1]
    if "```" in normalized:
        prefix = normalized.split("```", 1)[0].rstrip()
        if prefix:
            return prefix
    return normalized


def is_full_lean_document(text: str) -> bool:
    return bool(re.match(r"^\s*(import|namespace|open|set_option|theorem|lemma|example)\b", str(text)))


def extract_declaration_echo_body(text: str) -> str:
    normalized = normalize_generation_text(text).strip()
    if not re.match(r"^\s*(theorem|lemma|example)\b", normalized):
        return ""
    if re.match(r"^\s*import\b", normalized):
        return ""
    match = re.search(r":=\s*by\b", normalized)
    if not match:
        return ""
    body = normalized[match.end() :]
    body = textwrap.dedent(body).lstrip("\n")
    return body.rstrip()


def normalize_proof_completion_body(proof_text: str) -> str:
    proof = extract_candidate_completion(str(proof_text))
    echoed_body = extract_declaration_echo_body(proof)
    if echoed_body:
        return echoed_body
    if re.match(r"^\s*by\b", proof):
        body = re.sub(r"^\s*by\b", "", proof, count=1)
        body = textwrap.dedent(body).lstrip("\n")
        return body.rstrip()
    return proof.rstrip()


def _has_long_token_repetition(text: str, min_run: int = 6) -> bool:
    tokens = re.findall(r"\S+", text)
    if not tokens:
        return False
    run = 1
    for prev, current in zip(tokens, tokens[1:]):
        if current == prev:
            run += 1
            if run >= min_run:
                return True
        else:
            run = 1
    return False


def assess_completion_health(raw_text: str) -> dict[str, Any]:
    normalized_raw = normalize_generation_text(raw_text)
    extracted = extract_candidate_completion(normalized_raw)
    proof_body = normalize_proof_completion_body(extracted)
    issues: list[str] = []

    if PROVER_PROMPT_HEADER.lower() in normalized_raw.lower():
        issues.append("prompt_echo")
    if extract_declaration_echo_body(extracted):
        issues.append("declaration_echo")
    if not proof_body.strip():
        issues.append("empty")
    if re.search(r"[-_]{8,}", proof_body) or re.search(r"[-_]{8,}", normalized_raw):
        issues.append("punctuation_collapse")
    if _has_long_token_repetition(proof_body):
        issues.append("repetition")

    healthy = not issues
    return {
        "healthy": healthy,
        "issues": issues,
        "extracted": extracted,
        "proof_body": proof_body,
    }
