#!/bin/bash

# SmolVLA Model Evaluation Script
# Evaluates a trained SmolVLA policy on SO100 pick-and-place task

# Configuration
POLICY_PATH="/workspace/outputs/train/so100_grab_pen_smolvla_120k/checkpoints/090000/pretrained_model"
NUM_EPISODES=1
VIDEO_DIR="./eval_videos/90000"
GPU_ID=6

# Run evaluation with random cube positions and video recording
# CUDA_VISIBLE_DEVICES=${GPU_ID} python3 eval_smolvla.py \
#   --policy ${POLICY_PATH} \
#   --task SO100PickCubeBase-v0 \
#   --num-episodes ${NUM_EPISODES} \
#   --device cuda \
#   --randomize-cube \
#   --save-video \
#   --video-dir ${VIDEO_DIR} \
#   --render
  
CUDA_VISIBLE_DEVICES=${GPU_ID} python3 eval_smolvla.py \
  --policy ${POLICY_PATH} \
  --task SO100GrabPenBase-v0 \
  --num-episodes ${NUM_EPISODES} \
  --device cuda \
  --save-video \
  --video-dir ${VIDEO_DIR} \
  --render
