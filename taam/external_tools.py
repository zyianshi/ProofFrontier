from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Dict, Optional


def run_command_template(
    command_template: str,
    placeholders: Dict[str, str],
    timeout_sec: int,
) -> subprocess.CompletedProcess[str]:
    command = command_template.format(**placeholders)
    return subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
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
