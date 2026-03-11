#!/usr/bin/env bash
set -euo pipefail
OUT_ROOT="${TAAM_OUTPUT_ROOT:-/home/xuanxuan_awe/GenAI/taam_outputs_user}"
MATHLIB_ROOT="${MATHLIB_ROOT:-/home/xuanxuan_awe/deps/mathlib4_v420}"
mkdir -p "$OUT_ROOT"
exec > "$OUT_ROOT/mathlib4_v420_subset_100.log" 2>&1
source "$HOME/.elan/env"
source /home/xuanxuan_awe/miniconda3/etc/profile.d/conda.sh
conda activate work
export TAAM_WORK_PKGS=/home/xuanxuan_awe/.local/taam_work_pkgs
export PYTHONPATH="$TAAM_WORK_PKGS:${PYTHONPATH:-}"
cd /home/xuanxuan_awe/GenAI
python experiments/trace_mathlib_subset_with_leandojo.py \
  --mathlib-root "$MATHLIB_ROOT" \
  --include 'Algebra/**/*.lean' \
  --theorem-limit 100 \
  --skip-cache-get \
  --inventory-jsonl "$OUT_ROOT/mathlib4_v420_subset_inventory.jsonl" \
  --selected-files-json "$OUT_ROOT/mathlib4_v420_subset_selected_files.json"
