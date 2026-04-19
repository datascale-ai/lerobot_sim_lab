#!/bin/bash
# SmolVLA 训练示例（accelerate + LeRobot lerobot_train.py）。
# 默认: 仓库根 = 本脚本所在仓库；训练脚本从已安装 lerobot 包解析；权重在 third_party/smolvla_base
#
# 环境变量（均可选）:
#   LEROBOT_TRAIN  — 直接指定 lerobot_train.py 绝对路径
#   LEROOT         — 若使用本地 LeRobot 源码树，指向其根目录（将使用 LEROOT/src/lerobot/scripts/lerobot_train.py）
#   SMOLVLA_BASE   — 预训练权重目录，默认 REPO_ROOT/third_party/smolvla_base
#   DATA_GRAB_ROOT — 数据集根，默认 REPO_ROOT/data_grab_pen
#   CUDA_VISIBLE_DEVICES

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ -n "${LEROBOT_TRAIN:-}" ]]; then
  :
elif [[ -n "${LEROOT:-}" ]]; then
  LEROBOT_TRAIN="${LEROOT}/src/lerobot/scripts/lerobot_train.py"
  [[ -f "${LEROBOT_TRAIN}" ]] || {
    echo "missing ${LEROBOT_TRAIN} (check LEROOT)" >&2
    exit 1
  }
else
  LEROBOT_TRAIN="$(python3 <<'PY'
import pathlib
import sys

try:
    import lerobot as l
except ImportError:
    print("pip install lerobot (see README [lerobot] extra)", file=sys.stderr)
    sys.exit(1)
p = pathlib.Path(l.__file__).resolve().parent / "scripts" / "lerobot_train.py"
if not p.is_file():
    print(f"missing {p}", file=sys.stderr)
    sys.exit(1)
print(p)
PY
)"
fi

SMOLVLA_BASE="${SMOLVLA_BASE:-${REPO_ROOT}/third_party/smolvla_base}"
DATA_GRAB_ROOT="${DATA_GRAB_ROOT:-${REPO_ROOT}/data_grab_pen}"

accelerate launch --num_processes 1 \
  "${LEROBOT_TRAIN}" \
  --policy.path="${SMOLVLA_BASE}" \
  --policy.device=cuda \
  --policy.repo_id=local/so100_smolvla \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/so100_grab_pen_smolvla \
  --dataset.root="${DATA_GRAB_ROOT}" \
  --dataset.video_backend=torchcodec \
  --env.type=gym_manipulator \
  --env.task=SO100GrabPenBase-v0 \
  --batch_size=80 \
  --steps=120000 \
  --eval_freq=6000 \
  --save_freq=6000 \
  --num_workers=0 \
  --wandb.enable=true \
  --output_dir="${REPO_ROOT}/outputs/train/so100_grab_pen_smolvla_120k" \
  --policy.optimizer_lr=1e-4 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=120000 \
  --policy.scheduler_decay_lr=2.5e-6 \
  --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'
