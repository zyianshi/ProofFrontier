#!/usr/bin/env bash
set -euo pipefail
OUT_ROOT="${TAAM_OUTPUT_ROOT:-/home/xuanxuan_awe/GenAI/taam_outputs_user}"
MATHLIB_ROOT="${MATHLIB_ROOT:-/home/xuanxuan_awe/deps/mathlib4_v420}"
mkdir -p "$OUT_ROOT"
exec > "$OUT_ROOT/mathlib4_v420_build_graphs_100.log" 2>&1
source "$HOME/.elan/env"
source /home/xuanxuan_awe/miniconda3/etc/profile.d/conda.sh
conda activate work
export TAAM_WORK_PKGS=/home/xuanxuan_awe/.local/taam_work_pkgs
export PYTHONPATH="$TAAM_WORK_PKGS:${PYTHONPATH:-}"
cd /home/xuanxuan_awe/GenAI
python experiments/build_mathlib_trace_corpus.py \
  --traced-repo-root "$MATHLIB_ROOT" \
  --inventory-jsonl "$OUT_ROOT/mathlib4_v420_subset_inventory.jsonl" \
  --out-dir "$OUT_ROOT/mathlib4_v420_subset_graphs" \
  --module-prefix Mathlib.Algebra \
  --limit 100 \
  --max-depth 2 \
  --max-nodes 128 \
  --no-build-deps
