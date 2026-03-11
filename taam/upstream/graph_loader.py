from __future__ import annotations

from pathlib import Path

from .demo_data import build_demo_graph
from .extraction import FormalGraphExtractor
from .lean_trace import LeanTraceExtractor
from ..types import TAAMGraph


def load_graph(graph_json: str = "", lean_trace_json: str = "") -> TAAMGraph:
    if lean_trace_json:
        return LeanTraceExtractor.from_json(Path(lean_trace_json))
    if graph_json:
        return FormalGraphExtractor.from_json(Path(graph_json))
    return build_demo_graph()
