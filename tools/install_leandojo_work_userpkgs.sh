#!/usr/bin/env bash
set -euo pipefail
source /home/xuanxuan_awe/miniconda3/etc/profile.d/conda.sh
conda activate work
export TAAM_WORK_PKGS=/home/xuanxuan_awe/.local/taam_work_pkgs
mkdir -p "$TAAM_WORK_PKGS"
python -m pip install --target "$TAAM_WORK_PKGS" 'git+https://github.com/lean-dojo/LeanDojo.git'
PYTHONPATH="$TAAM_WORK_PKGS:${PYTHONPATH:-}" python - <<'PY'
import lean_dojo
print(lean_dojo.__file__)
print('LEAN_DOJO_OK')
PY
