#!/bin/bash
# 启动 LeRobot Policy Server。依赖已 pip 安装的 lerobot，无需 cd 到固定目录。
# 可选: export CUDA_VISIBLE_DEVICES=0

set -euo pipefail

python3 -m lerobot.async_inference.policy_server \
  --host=0.0.0.0 \
  --port=8080 \
  --fps=30 \
  --inference_latency=0.033 \
  --obs_queue_timeout=2
