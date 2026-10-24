#!/usr/bin/env bash
set -euo pipefail

# Execute in a native DSW Terminal. The driver prepares isolated auxiliary
# dependencies and runs the requested phase (use --phase full to skip pilots).
project=/mnt/workspace/3dmood
driver=/mnt/workspace/venvs/3d-mood-ppu-clean/bin/python
cd "$project"
export PYTHONPATH="$project${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export XFORMERS_DISABLED=1
export MPLBACKEND=Agg
exec "$driver" -B -m ioutest.unipose_suite --prepare "$@"
