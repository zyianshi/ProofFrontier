# TAAM Experiments

## Single run
```bash
python experiments/midstream_taam_generation/run_from_config.py --config configs/midstream_taam_generation/default_experiment.json
```

## Single run with DeepSeek prover attack
```bash
python experiments/midstream_taam_generation/run_from_config.py --config configs/midstream_taam_generation/deepseek_prover_experiment.json
```

Both configs use Lean-only graph/problem generation. They are now wired for a local `DeepSeek-Prover-V2-7B` backend and a local Lean validator.

Expected local setup:
```bash
python tools/bootstrap_lean_project.py --project-root lean_project --project-name TaamLean
```

The prover configs assume:
```bash
python tools/run_deepseek_prover.py --model-id deepseek-ai/DeepSeek-Prover-V2-7B --load-in-4bit ...
python tools/run_lean_validator.py --project-root lean_project ...
```

If you prefer an external backend instead of local HF inference, set:
```bash
$env:DEEPSEEK_PROVER_BACKEND_COMMAND = "your prover command with {lean_file} {out_json} {timeout_sec}"
$env:LEAN_VALIDATOR_BACKEND_COMMAND = "your lean check command with {lean_file} {out_json} {timeout_sec}"
```

## Sweep run
```bash
python experiments/midstream_taam_generation/sweep.py --config configs/midstream_taam_generation/sweep_small.json
```

## Sweep baseline: random mask
```bash
python experiments/midstream_taam_generation/sweep.py --config configs/midstream_taam_generation/sweep_random_mask.json
```

## Sweep baseline: low centrality first
```bash
python experiments/midstream_taam_generation/sweep.py --config configs/midstream_taam_generation/sweep_low_centrality.json
```

## Analyze sweep summaries
```bash
python experiments/analyze_results.py \
  --summary-csv artifacts/midstream_taam_generation/sweeps/demo_sweep_small_summary.csv \
  --summary-csv artifacts/midstream_taam_generation/sweeps_random/demo_sweep_random_mask_summary.csv \
  --out-json artifacts/midstream_taam_generation/analysis.json
```

## Convert Lean trace to canonical graph JSON
```bash
python experiments/convert_lean_trace_to_graph.py \
  --lean-trace-json configs/upstream_formal_graph/lean_trace_demo.json \
  --out-graph-json artifacts/upstream_formal_graph/converted_graph.json
```

## Batch index theorem declarations from Mathlib4
This step builds a source-level theorem inventory from a local `Mathlib4` checkout. It does not extract proof graphs by itself.

```bash
python experiments/index_mathlib_theorems.py \
  --mathlib-root path/to/Mathlib4 \
  --include Algebra/**/*.lean \
  --out-jsonl artifacts/upstream_formal_graph/mathlib4_algebra_theorems.jsonl
```

Use this inventory as the candidate theorem pool. Full TAAM graph extraction still needs a trace/export step from Lean/LeanDojo or your own compiler instrumentation.

## Real upstream with LeanDojo
The actual upstream should be:
1. trace `Mathlib4` with LeanDojo
2. export theorem inventory from the traced repo
3. build TAAM graph JSON from named theorem dependencies

Trace Mathlib4 and export theorem inventory:

```bash
python experiments/trace_mathlib_with_leandojo.py \
  --local-repo-path path/to/Mathlib4 \
  --traced-repo-root artifacts/upstream_formal_graph/mathlib4_traced \
  --inventory-jsonl artifacts/upstream_formal_graph/mathlib4_traced_theorems.jsonl
```

Then build TAAM graph corpus:

```bash
python experiments/build_mathlib_trace_corpus.py \
  --traced-repo-root artifacts/upstream_formal_graph/mathlib4_traced \
  --inventory-jsonl artifacts/upstream_formal_graph/mathlib4_traced_theorems.jsonl \
  --out-dir artifacts/upstream_formal_graph/mathlib4_taam_graphs \
  --module-prefix Mathlib.Algebra \
  --max-depth 2 \
  --max-nodes 128
```

The exported `*.graph.json` files are compatible with the current TAAM pipeline through `graph_json`.

## Main entry (also supports config)
```bash
python run.py --config configs/default_experiment.json
```

If both `graph_json` and `lean_trace_json` are empty, scripts use the built-in demo graph.

## Downstream export for prover training
Use TAAM hard samples as downstream training data:

```bash
python experiments/downstream_prover_eval/export_downstream_dataset.py --config configs/downstream_prover_eval/downstream_rl_minif2f.json
```

For SFT, the hard sample JSON must contain `proof_completion`. The demo trace now includes one, but your real pipeline must extract this from the original Lean proof or trace.

The intended SFT semantics are:
- prompt: TAAM-masked Lean problem
- target: the original full proof of the target theorem
- source of supervision: the complete proof chain before masking

The sample JSON now keeps:
- `lean_problem`: masked problem
- `full_lean_problem`: unmasked problem with the full visible proof context
- `proof_completion`: ground-truth proof target
- `proof_source`: where that proof came from

## Full downstream study: export -> train -> benchmark compare
```bash
python experiments/downstream_prover_eval/run_downstream_study.py --config configs/downstream_prover_eval/downstream_rl_gar_minif2f.json
```

This writes:
- `dataset/train.jsonl`, `val.jsonl`, `test.jsonl`
- `training_result.json`
- `benchmark_before.json`
- `benchmark_after.json`
- `downstream_summary.json`

The default downstream configs use mock trainer and mock benchmark scripts so the control flow can be tested before installing a real trainer or miniF2F runner.

## Compare benchmark runs directly
```bash
python experiments/compare_benchmark_results.py \
  --before artifacts/downstream_prover_eval/downstream_rl_minif2f/benchmark_before.json \
  --after artifacts/downstream_prover_eval/downstream_rl_minif2f/benchmark_after.json \
  --out-json artifacts/downstream_prover_eval/downstream_rl_minif2f/benchmark_comparison.json
```

## Prepare Lean4 miniF2F from GAR data
Use the Lean4 `miniF2F` data under `GAR-Official/data`.

```bash
python experiments/prepare_minif2f.py \
  --data-path path/to/GAR-Official/data \
  --split test \
  --out-jsonl artifacts/downstream_prover_eval/miniF2F_test_manifest.jsonl
```

## Run miniF2F through a prover command
```bash
python tools/run_minif2f_benchmark.py \
  --manifest-jsonl artifacts/downstream_prover_eval/miniF2F_test_manifest.jsonl \
  --split test \
  --model-ref deepseek-ai/DeepSeek-Prover-V2-7B \
  --out-json artifacts/downstream_prover_eval/miniF2F_test_results.json \
  --command-template "python tools/run_deepseek_prover.py --mock --lean-file \"{lean_file}\" --out-json \"{result_json}\""
```
