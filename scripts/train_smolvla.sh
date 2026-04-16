#!/bin/bash
# SmolVLA Training Script with Offline Mode

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=5

accelerate launch --num_processes 1 \
  lerobot/src/lerobot/scripts/lerobot_train.py \
  --policy.path=/workspace/smolvla_base \
  --policy.device=cuda \
  --policy.repo_id=local/so100_smolvla \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/so100_grab_pen_smolvla \
  --dataset.root=./data_grab_pen \
  --dataset.video_backend=torchcodec \
  --env.type=gym_manipulator \
  --env.task=SO100GrabPenBase-v0 \
  --batch_size=80 \
  --steps=120000 \
  --eval_freq=6000 \
  --save_freq=6000 \
  --num_workers=0 \
  --wandb.enable=true \
  --output_dir=outputs/train/so100_grab_pen_smolvla_120k \
  --policy.optimizer_lr=1e-4 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=120000 \
  --policy.scheduler_decay_lr=2.5e-6 \
  --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'

