from __future__ import annotations

import re
from typing import List

from ..types import ProblemState


def _safe_ident(raw: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]", "_", raw)
    if not s:
        s = "h"
    if s[0].isdigit():
        s = f"h_{s}"
    return s


def build_lean4_problem_from_state(state: ProblemState, include_import_mathlib: bool = True) -> str:
    """
    Build a Lean4 theorem skeleton directly from visible nodes + target.
    Assumes statements are Lean-compatible expressions.
    """
    lines: List[str] = []
    imports = state.imports or (["Mathlib"] if include_import_mathlib else [])
    for imp in imports:
        lines.append(f"import {imp}")
    if imports:
        lines.append("")

    theorem_name = _safe_ident(f"taam_{state.theorem_id}_{state.target_id}")
    context_chunks = [chunk.strip() for chunk in state.theorem_context if chunk.strip()]
    hyp_chunks = []
    for node_id, stmt in zip(state.visible_node_ids, state.visible_statements):
        hname = _safe_ident(f"h_{node_id}")
        hyp_chunks.append(f"({hname} : {stmt})")
    sig_chunks = context_chunks + hyp_chunks
    hyp_sig = " ".join(sig_chunks)

    lines.append(f"theorem {theorem_name} {hyp_sig} : {state.target_statement} := by")
    lines.append("  sorry")
    return "\n".join(lines)
