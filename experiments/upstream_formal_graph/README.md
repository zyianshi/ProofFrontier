# Upstream Formal Graph README

## Scope
This directory documents the **real upstream** that is already working in this repository:

`Mathlib4 -> LeanDojo tracing -> theorem inventory -> TAAM graph json`

The currently validated path is the **small-batch subset tracing route**, not full-repo tracing.

Validated target:
- Mathlib4 commit: `c211948581bde9846a99e32d97a03f0d5307c31e`
- Lean toolchain: `leanprover/lean4:v4.20.0`
- LeanDojo: installed into `~/.local/taam_work_pkgs` and injected through `PYTHONPATH`
- Current traced subset: `Mathlib.Algebra` small batch

Current validated output:
- inventory: `artifacts/upstream_formal_graph/mathlib4_v420_subset/mathlib4_v420_subset_inventory.jsonl`
- graphs: `artifacts/upstream_formal_graph/mathlib4_v420_subset/mathlib4_v420_subset_graphs/`
- successful graph count: `46`

## Directory Roles
- `trace_mathlib_subset_with_leandojo.py`: validated small-batch tracing entrypoint.
- `build_mathlib_trace_corpus.py`: converts traced inventory into TAAM graph JSON.
- `trace_mathlib_with_leandojo.py`: full traced-repo entrypoint; kept for future larger-scale runs.
- `run_mathlib4_upstream_wsl.py`: WSL wrapper for the full traced-repo route.
- `index_mathlib_theorems.py`: source-level theorem inventory without LeanDojo tracing.
- `convert_lean_trace_to_graph.py`: converts prepared Lean trace JSON into TAAM graph JSON.

## Environment Assumptions
Use the current mainline environment only:
- conda env: `work`
- Lean toolchain: `v4.20.0`
- active Mathlib worktree: `./deps/mathlib4_v420_user`

Required directories that should already exist after setup:
- `./deps/mathlib4`
- `./deps/mathlib4_v420_user`
- `./.elan`
- `./.local/taam_work_pkgs`

## 1. Enter WSL and activate the environment
From PowerShell:

```powershell
wsl -u <your wsl>
```

Then in WSL:

```bash
cd ./GenAI
source "$HOME/.elan/env"
source ./miniconda3/etc/profile.d/conda.sh
conda activate work
export TAAM_WORK_PKGS=./.local/taam_work_pkgs
export PYTHONPATH="$TAAM_WORK_PKGS:${PYTHONPATH:-}"
```

Sanity checks:

```bash
lean --version
python -c "import lean_dojo; print(lean_dojo.__file__)"
```

Expected:
- Lean is `v4.20.0`
- `lean_dojo` imports from `~/.local/taam_work_pkgs`

## 2. If LeanDojo is missing, install it into the user package target

```bash
bash tools/install_leandojo_work_userpkgs.sh
```

This does **not** use a venv. It installs LeanDojo into:
- `./.local/taam_work_pkgs`

## 3. Prepare the Mathlib4 worktree
Current working tree for the validated upstream:
- `./deps/mathlib4_v420_user`

If it is missing, recreate it from the main repo:

```bash
git -C ./deps/mathlib4 worktree add --detach ./deps/mathlib4_v420_user c211948581bde9846a99e32d97a03f0d5307c31e
```

Then populate the cache:

```bash
cd ./deps/mathlib4_v420_user
source "$HOME/.elan/env"
lake exe cache get
```

## 4. Reproduce the validated small-batch subset tracing
This is the path that has already been verified in this repository.

```bash
cd ./GenAI
source "$HOME/.elan/env"
source ./miniconda3/etc/profile.d/conda.sh
conda activate work
export TAAM_WORK_PKGS=./.local/taam_work_pkgs
export PYTHONPATH="$TAAM_WORK_PKGS:${PYTHONPATH:-}"

python experiments/upstream_formal_graph/trace_mathlib_subset_with_leandojo.py \
  --mathlib-root ./deps/mathlib4_v420_user \
  --include 'Algebra/**/*.lean' \
  --theorem-limit 100 \
  --skip-cache-get \
  --inventory-jsonl ./GenAI/artifacts/upstream_formal_graph/mathlib4_v420_subset/mathlib4_v420_subset_inventory.jsonl \
  --selected-files-json ./GenAI/artifacts/upstream_formal_graph/mathlib4_v420_subset/mathlib4_v420_subset_selected_files.json
```

What this does:
- scans Mathlib source files under `Mathlib/Algebra/**/*.lean`
- chooses enough files to cover the theorem limit
- copies LeanDojo `ExtractData.lean` into the worktree
- patches one brittle assertion in the extractor
- runs file-level tracing
- exports theorem inventory from the traced files

Expected output files:
- `artifacts/upstream_formal_graph/mathlib4_v420_subset/mathlib4_v420_subset_inventory.jsonl`
- `artifacts/upstream_formal_graph/mathlib4_v420_subset/mathlib4_v420_subset_selected_files.json`

## 5. Build TAAM graphs from the traced inventory
Use the currently validated inventory-driven graph builder:

```bash
cd ./GenAI
source "$HOME/.elan/env"
source ./miniconda3/etc/profile.d/conda.sh
conda activate work
export TAAM_WORK_PKGS=./.local/taam_work_pkgs
export PYTHONPATH="$TAAM_WORK_PKGS:${PYTHONPATH:-}"

python experiments/upstream_formal_graph/build_mathlib_trace_corpus.py \
  --traced-repo-root ./deps/mathlib4_v420_user \
  --inventory-jsonl ./GenAI/artifacts/upstream_formal_graph/mathlib4_v420_subset/mathlib4_v420_subset_inventory.jsonl \
  --out-dir ./GenAI/artifacts/upstream_formal_graph/mathlib4_v420_subset/mathlib4_v420_subset_graphs \
  --module-prefix Mathlib.Algebra \
  --limit 100 \
  --max-depth 2 \
  --max-nodes 128 \
  --no-build-deps
```

Why `--no-build-deps` is recommended for the validated path:
- subset-traced repos do not reload from disk as reliably as the in-memory traced objects
- this mode constructs graphs directly from the exported inventory and its `premise_full_names`
- this is the route that successfully produced the current `46` graphs

Expected output:
- graph directory: `artifacts/upstream_formal_graph/mathlib4_v420_subset/mathlib4_v420_subset_graphs/`
- manifest: `artifacts/upstream_formal_graph/mathlib4_v420_subset/mathlib4_v420_subset_graphs/manifest.jsonl`

## 6. Inspect the current validated outputs
Quick checks:

```bash
wc -l artifacts/upstream_formal_graph/mathlib4_v420_subset/mathlib4_v420_subset_inventory.jsonl
wc -l artifacts/upstream_formal_graph/mathlib4_v420_subset/mathlib4_v420_subset_graphs/manifest.jsonl
head -n 5 artifacts/upstream_formal_graph/mathlib4_v420_subset/mathlib4_v420_subset_graphs/manifest.jsonl
```

Current validated result in this repository:
- inventory lines: `46`
- graph manifest lines: `46`
- exported graphs: `46/46`

## 7. Feed the graphs into the TAAM midstream
One graph can be passed directly into the current TAAM pipeline:

```bash
python run.py --graph-json ./GenAI/artifacts/upstream_formal_graph/mathlib4_v420_subset/mathlib4_v420_subset_graphs/vsub_self.graph.json
```

## Optional: source-level theorem indexing without LeanDojo tracing
If you only want a theorem pool first:

```bash
python experiments/upstream_formal_graph/index_mathlib_theorems.py \
  --mathlib-root ./deps/mathlib4_v420_user \
  --include 'Algebra/**/*.lean' \
  --out-jsonl ./GenAI/artifacts/upstream_formal_graph/mathlib4_algebra_theorems.jsonl
```

## Optional: full traced-repo route
There is also a full traced-repo route:

```bash
python experiments/upstream_formal_graph/run_mathlib4_upstream_wsl.py \
  --config configs/upstream_formal_graph/mathlib4_leandojo_upstream.json
```

This route is kept for future larger-scale tracing, but the path that has already been validated end-to-end in this repository is the subset route documented above.

## Known limitations
1. Current graphs are theorem-level dependency graphs, not tactic-state/subgoal graphs.
2. The currently validated output is a small `Mathlib.Algebra` subset, not full Mathlib4.
3. For subset tracing, inventory-driven graph construction is more robust than reloading a traced repo from disk.
4. Some `premise_full_names` refer to non-theorem symbols; these are skipped during graph construction.

## Minimal reproduction checklist
1. Activate `conda work`.
2. Load `~/.elan/env`.
3. Export `PYTHONPATH=./.local/taam_work_pkgs`.
4. Ensure `./deps/mathlib4_v420_user` exists and `lake exe cache get` has completed.
5. Run `trace_mathlib_subset_with_leandojo.py`.
6. Run `build_mathlib_trace_corpus.py --no-build-deps`.
7. Verify `46` graphs under `artifacts/upstream_formal_graph/mathlib4_v420_subset/mathlib4_v420_subset_graphs/`.
