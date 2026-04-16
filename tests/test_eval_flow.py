"""Tests for evaluation flow — environment creation, reset, step with random policy."""

import numpy as np
import pytest

import lerobot_sim_lab.envs

lerobot_sim_lab.envs.register_envs()


@pytest.fixture
def pick_cube_env():
    """Create a base SO-100 environment for eval testing."""
    from gymnasium.wrappers import TimeLimit

    from lerobot_sim_lab.envs.so100_gym_env import SO100PickCubeGymEnv

    env = SO100PickCubeGymEnv(control_dt=1 / 30, render_mode="rgb_array")
    env = TimeLimit(env, max_episode_steps=200)
    yield env
    env.close()


class TestRandomPolicy:
    def test_env_reset(self, pick_cube_env):
        obs, info = pick_cube_env.reset(seed=42)
        assert isinstance(obs, dict)
        assert "state" in obs

    def test_random_episode(self, pick_cube_env):
        obs, _ = pick_cube_env.reset(seed=42)
        total_reward = 0.0
        steps = 0
        terminated = truncated = False

        while not (terminated or truncated) and steps < 50:
            action = pick_cube_env.action_space.sample()
            obs, reward, terminated, truncated, info = pick_cube_env.step(action)
            total_reward += reward
            steps += 1

        assert steps > 0
        assert isinstance(total_reward, (float, int, np.floating))


class TestVectorEnv:
    def test_sync_vector_env(self):
        from gymnasium.vector import SyncVectorEnv

        from lerobot_sim_lab.envs.so100_gym_env import SO100PickCubeGymEnv

        def make_env():
            return SO100PickCubeGymEnv(control_dt=1 / 30, render_mode="rgb_array")

        n_envs = 2
        vec_env = SyncVectorEnv([make_env for _ in range(n_envs)])
        try:
            obs, info = vec_env.reset(seed=[100, 101])
            assert isinstance(obs, dict)

            actions = vec_env.action_space.sample()
            obs, rewards, terminated, truncated, info = vec_env.step(actions)
            assert rewards.shape == (n_envs,)
        finally:
            vec_env.close()
