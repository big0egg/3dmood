#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
experiment_root=/root/robot-realism/agentview_prompt_v2
runtime_python=/tmp/robot-realism/cosmos25/venv/bin/python

if [[ ! -x "$runtime_python" ]]; then
    printf '缺少生成环境：%s\n' "$runtime_python" >&2
    exit 1
fi

if [[ "${1:-}" == --background ]]; then
    shift
    # Keep the launcher deliberately fixed to the approved /root experiment.
    if [[ $# -ne 0 ]]; then
        printf '后台入口使用已确认的默认配置；自定义参数请使用前台入口。\n' >&2
        exit 1
    fi
    mkdir -p "$experiment_root/runs"
    nohup bash "$project_dir/scripts/run_agentview_prompt_full.sh" \
        >> "$experiment_root/runs/launch.log" 2>&1 < /dev/null &
    task_pid=$!
    printf '%s\n' "$task_pid" > "$experiment_root/runs/launcher.pid"
    printf '已提交全量启动，PID=%s\n日志：%s\n' "$task_pid" "$experiment_root/runs/launch.log"
    exit 0
fi

cd "$project_dir"
export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export MPLCONFIGDIR=/root/robot-realism/cosmos25/cache/matplotlib
exec "$runtime_python" -u -B -m robot_realism prompt-suite "$@"
