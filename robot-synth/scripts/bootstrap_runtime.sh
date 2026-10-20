#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
capture_python="${CAPTURE_PYTHON:-python3}"
capture_venv="${CAPTURE_VENV:-$project_dir/.venv}"
libero_ref="${LIBERO_REF:-master}"
robosuite_ref="${ROBOSUITE_REF:-v1.4.0}"

"$capture_python" -m venv "$capture_venv"
"$capture_venv/bin/python" -m pip install -r "$project_dir/requirements-runtime.txt"
"$capture_venv/bin/python" -m pip install --no-deps -e "$project_dir"

mkdir -p "$project_dir/third_party"
clone_assets() {
  local url="$1" directory="$2" reference="$3"
  if [[ -e "$directory" ]]; then
    if [[ ! -d "$directory/.git" ]]; then
      echo "Existing asset directory is not a checkout; set asset_roots in the config: $directory" >&2
      return 1
    fi
    echo "Using existing asset checkout without changing it: $directory"
  else
    git clone --filter=blob:none "$url" "$directory"
    git -C "$directory" checkout "$reference"
  fi
  git -C "$directory" rev-parse HEAD
}

clone_assets https://github.com/Lifelong-Robot-Learning/LIBERO.git "$project_dir/third_party/LIBERO" "$libero_ref"
clone_assets https://github.com/ARISE-Initiative/robosuite.git "$project_dir/third_party/robosuite" "$robosuite_ref"

"$capture_venv/bin/python" - <<'PY'
import ctypes.util
if not ctypes.util.find_library('OSMesa'):
    raise SystemExit('OSMesa is missing. Build the provided Dockerfile or install libosmesa6 in your dedicated capture image; do not alter the PPU model environment.')
print('Python dependencies/assets present. Run doctor, then the real native-renderer test.')
PY
