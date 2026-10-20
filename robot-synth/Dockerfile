FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    git libosmesa6 libgl1 ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /mnt/workspace/3dmood/robot-synth
COPY . .
ARG LIBERO_REF=master
ARG ROBOSUITE_REF=v1.4.0
RUN LIBERO_REF="$LIBERO_REF" ROBOSUITE_REF="$ROBOSUITE_REF" bash scripts/bootstrap_runtime.sh
ENV MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["/mnt/workspace/3dmood/robot-synth/.venv/bin/python", "-m", "robot_synth"]
