#!/bin/bash
# PolicyServer启动脚本

# export CUDA_VISIBLE_DEVICES=5  # 使用GPU 5（与训练一致）

cd /workspace/lerobot

python3 -m lerobot.async_inference.policy_server \
    --host=0.0.0.0 \
    --port=8080 \
    --fps=30 \
    --inference_latency=0.033 \
    --obs_queue_timeout=2

