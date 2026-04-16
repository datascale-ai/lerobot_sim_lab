"""Tests for SO-100 gymnasium environments."""

import gymnasium as gym
import numpy as np
import pytest

import lerobot_sim_lab.envs  # noqa: F401 — triggers env registration

# Ensure envs are registered
lerobot_sim_lab.envs.register_envs()


@pytest.fixture
def scripted_env():
    """Create and yield a scripted SO-100 environment."""
    env = gym.make("lerobot_sim_lab/SO100PickCubeScripted-v0")
    yield env
    env.close()


@pytest.fixture
def base_env():
    """Create and yield a base SO-100 environment."""
    env = gym.make("lerobot_sim_lab/SO100PickCube-v0")
    yield env
    env.close()


class TestEnvCreation:
    def test_scripted_env_creates(self):
        env = gym.make("lerobot_sim_lab/SO100PickCubeScripted-v0")
        assert env is not None
        env.close()

    def test_base_env_creates(self):
        env = gym.make("lerobot_sim_lab/SO100PickCube-v0")
        assert env is not None
        env.close()

    def test_legacy_gym_hil_id(self):
        env = gym.make("gym_hil/SO100PickCubeScripted-v0")
        assert env is not None
        env.close()


class TestEnvReset:
    def test_reset_returns_obs_and_info(self, scripted_env):
        obs, info = scripted_env.reset()
        assert isinstance(obs, dict)
        assert isinstance(info, dict)

    def test_obs_contains_state(self, scripted_env):
        obs, _ = scripted_env.reset()
        assert "state" in obs
        assert isinstance(obs["state"], np.ndarray)
        assert obs["state"].ndim == 1

    def test_reset_with_seed(self, scripted_env):
        obs1, _ = scripted_env.reset(seed=42)
        obs2, _ = scripted_env.reset(seed=42)
        np.testing.assert_array_equal(obs1["state"], obs2["state"])


class TestEnvStep:
    def test_step_returns_five_tuple(self, scripted_env):
        scripted_env.reset()
        action = scripted_env.unwrapped.get_scripted_action()
        result = scripted_env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, dict)
        assert isinstance(reward, (float, int, np.floating))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_multiple_steps(self, scripted_env):
        scripted_env.reset()
        for _ in range(5):
            action = scripted_env.unwrapped.get_scripted_action()
            obs, reward, terminated, truncated, info = scripted_env.step(action)
            if terminated or truncated:
                break

    def test_random_action_accepted(self, base_env):
        base_env.reset()
        action = base_env.action_space.sample()
        obs, reward, terminated, truncated, info = base_env.step(action)
        assert isinstance(obs, dict)


class TestFullEpisode:
    def test_episode_terminates(self, scripted_env):
        scripted_env.reset()
        total_steps = 0
        for _ in range(200):
            action = scripted_env.unwrapped.get_scripted_action()
            obs, reward, terminated, truncated, info = scripted_env.step(action)
            total_steps += 1
            if terminated or truncated:
                break
        assert total_steps > 0
