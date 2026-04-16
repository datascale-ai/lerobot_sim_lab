#!/bin/bash

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=5

accelerate launch --num_processes 1 \
  lerobot/src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=local/so100_grab_pen_pi0 \
  --dataset.root=./data_grab_pen \
  --dataset.video_backend=torchcodec \
  --env.type=gym_manipulator \
  --env.task=SO100GrabPenBase-v0 \
  --policy.type=pi0 \
  --policy.pretrained_path=/workspace/pi0_base \
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
  --output_dir=outputs/train/so100_grab_pen_pi0_3k


