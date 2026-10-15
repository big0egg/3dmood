#!/usr/bin/env bash
set -euo pipefail

project=/mnt/workspace/3dmood
driver=/mnt/workspace/venvs/3d-mood-ppu/bin/python
cd "$project"
export PYTHONPATH="$project${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export XFORMERS_DISABLED=1
export MPLBACKEND=Agg
exec "$driver" -B -m ioutest.hot3d.runner "$@"
