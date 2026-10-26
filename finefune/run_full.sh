#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${FINEFUNE_PYTHON:-/usr/local/bin/python3}" -B "$project_dir/workflow.py" "$@"
