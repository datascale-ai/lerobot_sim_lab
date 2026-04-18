#!/bin/bash
# Pi0 训练示例。路径约定同 train_smolvla.sh（仓库根、pip 解析 lerobot_train、third_party/pi0_base）。
#
# 环境变量: LEROBOT_TRAIN, LEROOT, PI0_BASE, DATA_GRAB_ROOT, CUDA_VISIBLE_DEVICES

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

PI0_BASE="${PI0_BASE:-${REPO_ROOT}/third_party/pi0_base}"
DATA_GRAB_ROOT="${DATA_GRAB_ROOT:-${REPO_ROOT}/data_grab_pen}"

accelerate launch --num_processes 1 \
  "${LEROBOT_TRAIN}" \
  --dataset.repo_id=local/so100_grab_pen_pi0 \
  --dataset.root="${DATA_GRAB_ROOT}" \
  --dataset.video_backend=torchcodec \
  --env.type=gym_manipulator \
  --env.task=SO100GrabPenBase-v0 \
  --policy.type=pi0 \
  --policy.pretrained_path="${PI0_BASE}" \
  --policy.repo_id=local/so100_pi0 \
  --policy.compile_model=true \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.device=cuda \
  --batch_size=32 \
  --steps=3000 \
  --job_name=pi0_training \
  --output_dir="${REPO_ROOT}/outputs/train/so100_grab_pen_pi0_3k"
