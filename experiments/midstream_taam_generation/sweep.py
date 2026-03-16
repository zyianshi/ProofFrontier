from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taam.config import load_sweep_config
from taam.upstream.graph_loader import load_graph
from taam.midstream.pipeline import run_experiment


def _resolve_manifest_graph_path(manifest_path: Path, graph_json: str) -> str:
    raw_path = Path(graph_json)
    if raw_path.exists():
        return str(raw_path)
    candidate = manifest_path.parent / raw_path.name
    if candidate.exists():
        return str(candidate)
    return graph_json


def _iter_graph_specs(cfg) -> List[Dict[str, str]]:
    specs: List[Dict[str, str]] = []
    if cfg.graph_json:
        specs.append({'graph_json': cfg.graph_json, 'label': Path(cfg.graph_json).stem})
    elif cfg.lean_trace_json:
        specs.append({'lean_trace_json': cfg.lean_trace_json, 'label': Path(cfg.lean_trace_json).stem})
    elif cfg.graph_manifest_jsonl:
        manifest_path = Path(cfg.graph_manifest_jsonl)
        with manifest_path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                graph_json = str(row.get('graph_json', '')).strip()
                if row.get('status') == 'ok' and graph_json:
                    resolved_graph_json = _resolve_manifest_graph_path(manifest_path, graph_json)
                    specs.append({'graph_json': resolved_graph_json, 'label': Path(resolved_graph_json).stem})
    elif cfg.graph_dir:
        for path in sorted(Path(cfg.graph_dir).glob(cfg.graph_glob)):
            specs.append({'graph_json': str(path), 'label': path.stem})
    else:
        raise SystemExit('Sweep requires one of graph_json, lean_trace_json, graph_manifest_jsonl, or graph_dir.')

    if cfg.graph_limit > 0:
        specs = specs[: cfg.graph_limit]
    if not specs:
        raise SystemExit('No graph inputs resolved for sweep.')
    return specs


def _safe_dir_name(text: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in text)


def _sample_row(sample, graph_spec: Dict[str, str], threshold: float, seed: int, out_dir: Path) -> Dict[str, object]:
    return {
        'sweep_name': '',
        'graph_label': graph_spec.get('label', ''),
        'graph_json': graph_spec.get('graph_json', ''),
        'lean_trace_json': graph_spec.get('lean_trace_json', ''),
        'threshold': threshold,
        'seed': seed,
        'solver_type': '',
        'masking_strategy': '',
        'found_hard_sample': sample is not None,
        'failed_on': '' if sample is None else sample.failed_on,
        'hidden_count': 0 if sample is None else len(sample.hidden_nodes),
        'hidden_lemma_count': 0 if sample is None else sample.hidden_lemma_count,
        'visible_lemma_count': 0 if sample is None else sample.visible_lemma_count,
        'well_posed': '' if sample is None else sample.well_posed,
        'theorem_id': '' if sample is None else sample.theorem_id,
        'target_id': '' if sample is None else sample.target_id,
        'theorem_domain': '' if sample is None else sample.theorem_domain,
        'source_file_path': '' if sample is None else sample.source_file_path,
        'proof_source': '' if sample is None else sample.proof_source,
        'source_graph_size': 0 if sample is None else sample.source_graph_size,
        'source_graph_edge_count': 0 if sample is None else sample.source_graph_edge_count,
        'sample_json': '' if sample is None else sample.sample_json_path,
        'out_dir': str(out_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Run TAAM hyper-parameter sweep')
    parser.add_argument('--config', type=str, default='configs/midstream_taam_generation/sweep_small.json')
    args = parser.parse_args()

    cfg = load_sweep_config(Path(args.config))
    graph_specs = _iter_graph_specs(cfg)

    out_root = Path(cfg.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / f'{cfg.name}_summary.csv'

    rows = []
    for graph_spec in graph_specs:
        graph = load_graph(
            graph_json=graph_spec.get('graph_json', ''),
            lean_trace_json=graph_spec.get('lean_trace_json', ''),
        )
        graph_label = _safe_dir_name(graph_spec.get('label', graph.theorem_id))
        for threshold in cfg.capability_thresholds:
            for seed in cfg.seeds:
                out_dir = out_root / graph_label / f'thr_{threshold:.2f}_seed_{seed}'
                sample = run_experiment(
                    graph.copy(),
                    out_dir,
                    threshold,
                    seed,
                    solver_type=cfg.solver_type,
                    solver_config=cfg.solver_config,
                    validation_config=cfg.validation_config,
                    masking_strategy=cfg.masking_strategy,
                    masking_config=cfg.masking_config,
                )
                row = _sample_row(sample, graph_spec, threshold, seed, out_dir)
                row['sweep_name'] = cfg.name
                row['solver_type'] = cfg.solver_type
                row['masking_strategy'] = cfg.masking_strategy
                rows.append(row)
                print(
                    f"graph={graph.theorem_id}, threshold={threshold:.2f}, seed={seed}, found={sample is not None}, "
                    f"failed_on={'' if sample is None else sample.failed_on}"
                )

    fieldnames = [
        'sweep_name',
        'graph_label',
        'graph_json',
        'lean_trace_json',
        'threshold',
        'seed',
        'solver_type',
        'masking_strategy',
        'found_hard_sample',
        'failed_on',
        'hidden_count',
        'hidden_lemma_count',
        'visible_lemma_count',
        'well_posed',
        'theorem_id',
        'target_id',
        'theorem_domain',
        'source_file_path',
        'proof_source',
        'source_graph_size',
        'source_graph_edge_count',
        'sample_json',
        'out_dir',
    ]
    with summary_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'Saved summary: {summary_path.resolve()}')


if __name__ == '__main__':
    main()
