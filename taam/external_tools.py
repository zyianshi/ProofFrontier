from __future__ import annotations

import json
import os
import re
import subprocess
import textwrap
from pathlib import Path
from typing import Dict, Optional


def run_command_template(
    command_template: str,
    placeholders: Dict[str, str],
    timeout_sec: int,
) -> subprocess.CompletedProcess[str]:
    command = command_template.format(**placeholders)
    if os.name == "nt":
        return subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    return subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )


def parse_json_result(path: Path, truthy_keys: tuple[str, ...]) -> Optional[bool]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    for key in truthy_keys:
        if key in data:
            return bool(data[key])
    return None


def parse_verdict_from_text(text: str, tag: str) -> Optional[bool]:
    pattern = rf"{re.escape(tag)}:\s*(PROVED|FAILED|PASSED|ERROR)"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    verdict = match.group(1).upper()
    return verdict in {"PROVED", "PASSED"}


def materialize_lean_completion(source_text: str, completion: str) -> str:
    source = str(source_text)
    proof = str(completion).strip()
    if not proof:
        return source
    if re.match(r"^\s*(import|theorem|lemma|example)\b", proof):
        return proof.rstrip() + "\n"
    if re.match(r"^\s*by\b", proof):
        body = re.sub(r"^\s*by\b", "", proof, count=1)
        body = textwrap.dedent(body).lstrip("\n")
        if "sorry" in source:
            body_text = body or "skip"
            match = re.search(r"(?m)^(?P<indent>\s*)sorry\b", source)
            if match:
                indent = match.group("indent")
                indented = "\n".join((f"{indent}{line}" if line.strip() else line) for line in body_text.splitlines())
                return source[: match.start()] + indented + source[match.end() :]
            return source.replace("sorry", body_text, 1)
        if ":= by" in source:
            prefix, _suffix = source.rsplit(":= by", 1)
            return f"{prefix}:= {proof}\n"
    if "sorry" not in source:
        return f"{source.rstrip()}\n\n{proof}\n"
    return source.replace("sorry", proof, 1)
