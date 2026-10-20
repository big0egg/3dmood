#!/usr/bin/env bash
set -euo pipefail
data_env=/mnt/workspace/venvs/hot3d-data
if [[ ! -x "$data_env/bin/python" ]]; then
    python3 -m venv "$data_env"
fi
if ! "$data_env/bin/python" -B -c 'from projectaria_tools.core import data_provider; import numpy,scipy,cv2,PIL,matplotlib' 2>/dev/null; then
    "$data_env/bin/python" -m pip install 'numpy==1.26.4' 'opencv-python-headless==4.11.0.86' 'scipy>=1.13,<2' 'Pillow>=10,<13' 'matplotlib>=3.8,<4' projectaria-tools
fi
"$data_env/bin/python" -m pip check
"$data_env/bin/python" -m pip freeze > "$data_env/hot3d-data-lock.txt"
