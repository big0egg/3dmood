#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bash "$project_dir/setup_data_env.sh"
exec /mnt/workspace/venvs/3d-mood-ppu/bin/python -B "$project_dir/run.py" run "$@"
