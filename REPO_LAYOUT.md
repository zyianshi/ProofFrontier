# Repository Layout

## Core Package
- `taam/upstream/`: Mathlib4 theorem indexing, LeanDojo tracing, graph extraction, and graph loading.
- `taam/midstream/`: topology profiling, masking, Lean problem construction, prover attack, and validation.
- `taam/downstream/`: dataset export, benchmark handling, and downstream evaluation utilities.
- `taam/config.py`, `taam/types.py`, `taam/external_tools.py`: shared config, shared data structures, and external command helpers.

## Stage Scripts
- `experiments/upstream_formal_graph/`: real upstream scripts for Mathlib4 and LeanDojo.
- `experiments/midstream_taam_generation/`: TAAM generation, sweep, and analysis scripts.
- `experiments/downstream_prover_eval/`: downstream dataset export, miniF2F prep, benchmark comparison, and study scripts.

## Stage Configs
- `configs/upstream_formal_graph/`
- `configs/midstream_taam_generation/`
- `configs/downstream_prover_eval/`

## Stage Artifacts
- `artifacts/upstream_formal_graph/`: traced Mathlib inventories and graph outputs.
- `artifacts/midstream_taam_generation/`: TAAM generation outputs and legacy midstream outputs.
- `artifacts/downstream_prover_eval/`: downstream training and benchmark outputs.

## Shared Entry Point
- `run.py`: top-level midstream experiment entry point.

## Shared Tooling
- `tools/`: shared Lean/prover/bootstrap helper scripts used across stages.
