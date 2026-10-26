#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
data_env="${FINEFUNE_DATA_ENV:-/mnt/workspace/venvs/finefune-data}"
if [[ ! -x "$data_env/bin/python" ]]; then
    python3 -m venv "$data_env"
fi
"$data_env/bin/python" -m pip install -r "$project_dir/requirements-data.txt"
"$data_env/bin/python" -m pip check
"$data_env/bin/python" -m pip freeze > "$data_env/finefune-data-lock.txt"
printf 'Data interpreter: %s\n' "$data_env/bin/python"
