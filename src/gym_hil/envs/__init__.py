"""Compatibility exports for older gym_hil imports."""

from lerobot_sim_lab.envs.so100_gym_env import SO100GrabPenGymEnv, SO100PickCubeGymEnv
from lerobot_sim_lab.envs.so100_scripted_env import SO100PickCubeScriptedEnv

__all__ = ["SO100GrabPenGymEnv", "SO100PickCubeGymEnv", "SO100PickCubeScriptedEnv"]
