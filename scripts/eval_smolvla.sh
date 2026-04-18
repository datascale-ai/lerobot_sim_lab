#!/bin/bash
# SmolVLA 评估示例。在仓库根下解析默认路径；可用环境变量覆盖: POLICY_PATH, VIDEO_DIR, GPU_ID
#
# 需已: pip install -e ".[lerobot]"

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

POLICY_PATH="${POLICY_PATH:-${REPO_ROOT}/outputs/train/so100_grab_pen_smolvla_120k/checkpoints/090000/pretrained_model}"
NUM_EPISODES="${NUM_EPISODES:-1}"
VIDEO_DIR="${VIDEO_DIR:-${REPO_ROOT}/eval_videos/90000}"
GPU_ID="${GPU_ID:-0}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 -m lerobot_sim_lab.evaluation.eval_smolvla \
  --policy "${POLICY_PATH}" \
  --task SO100GrabPenBase-v0 \
  --num-episodes "${NUM_EPISODES}" \
  --device cuda \
  --save-video \
  --video-dir "${VIDEO_DIR}" \
  --render
