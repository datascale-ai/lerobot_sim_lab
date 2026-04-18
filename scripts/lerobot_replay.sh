#!/bin/bash
# 回放 LeRobot 数据集中的 episode。默认路径相对于本仓库根目录（由脚本位置自动解析）。
# 可选环境变量: GPU_ID, LEROBOT_SIM_LAB_REPO_ROOT（覆盖仓库根）

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export LEROBOT_SIM_LAB_REPO_ROOT="${LEROBOT_SIM_LAB_REPO_ROOT:-${REPO_ROOT}}"

GPU_ID="${GPU_ID:-0}"
export MUJOCO_GL=glfw
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_OFFLINE=0

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 - <<'PY'
import os
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

repo_root = Path(os.environ["LEROBOT_SIM_LAB_REPO_ROOT"])

import lerobot_sim_lab.envs
lerobot_sim_lab.envs.register_envs()
from lerobot_sim_lab.config.scenarios.pen_grab import PEN_SCENARIOS

if "gym_gym_manipulator/SO100PickCubeBase-v0" not in gym.registry:
    gym.register(
        id="gym_gym_manipulator/SO100PickCubeBase-v0",
        entry_point="lerobot_sim_lab.envs.so100_gym_env:SO100PickCubeGymEnv",
        max_episode_steps=400,
        kwargs={"control_dt": 1 / 30},
    )

if "gym_gym_manipulator/SO100GrabPenBase-v0" not in gym.registry:
    gym.register(
        id="gym_gym_manipulator/SO100GrabPenBase-v0",
        entry_point="lerobot_sim_lab.envs.so100_gym_env:SO100GrabPenGymEnv",
        max_episode_steps=2000,
        kwargs={"control_dt": 1 / 30},
    )

from lerobot_sim_lab.envs.wrappers.viewer_wrapper import PassiveViewerWrapper
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION

task = "SO100GrabPenBase-v0"
dataset_repo_id = "local/so100_grab_pen_smolvla"
dataset_root = str(repo_root / "data_grab_pen")
episode = 59

env = gym.make(f"gym_gym_manipulator/{task}")
env = PassiveViewerWrapper(env, default_camera="camera_front_new", sync_every_n_steps=1)

info_path = Path(dataset_root) / "meta" / "info.json"
if not info_path.exists():
    raise SystemExit(f"missing dataset info: {info_path}")
info = __import__("json").loads(info_path.read_text())
total_episodes = int(info.get("total_episodes", 0))
if episode < 0 or episode >= total_episodes:
    raise SystemExit(f"episode out of range: {episode} (total_episodes={total_episodes})")

dataset = LeRobotDataset(dataset_repo_id, root=dataset_root, episodes=[episode])
episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == episode).sort("frame_index")
if len(episode_frames) == 0:
    raise SystemExit(f"no frames for episode {episode}")
actions = episode_frames.select_columns(ACTION)

scenario_ids = [scenario["id"] for scenario in PEN_SCENARIOS if scenario["id"] != 0]
scenario_id = scenario_ids[episode % len(scenario_ids)]
obs, info = env.reset(options={"scenario_idx": scenario_id})
for idx in range(len(episode_frames)):
    t0 = time.perf_counter()
    action_array = actions[idx][ACTION]
    action = np.array(action_array, dtype=np.float32)
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        break
    dt = time.perf_counter() - t0
    sleep_s = 1 / dataset.fps - dt
    if sleep_s > 0:
        time.sleep(sleep_s)

env.close()
PY

