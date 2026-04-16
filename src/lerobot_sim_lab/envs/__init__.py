"""Environment registrations for lerobot_sim_lab."""

from gymnasium.envs.registration import register, registry


def _register_once(env_id: str, entry_point: str) -> None:
    if env_id not in registry:
        register(id=env_id, entry_point=entry_point)


def register_envs() -> None:
    """Register project environments with gymnasium."""
    _register_once(
        "lerobot_sim_lab/SO100PickCube-v0",
        "lerobot_sim_lab.envs.so100_gym_env:SO100PickCubeGymEnv",
    )
    _register_once(
        "lerobot_sim_lab/SO100PickCubeScripted-v0",
        "lerobot_sim_lab.envs.so100_scripted_env:SO100PickCubeScriptedEnv",
    )
    _register_once(
        "lerobot_sim_lab/SO100GrabPen-v0",
        "lerobot_sim_lab.envs.so100_gym_env:SO100GrabPenGymEnv",
    )
    _register_once(
        "gym_hil/SO100PickCubeBase-v0",
        "lerobot_sim_lab.envs.so100_gym_env:SO100PickCubeGymEnv",
    )
    _register_once(
        "gym_hil/SO100PickCubeScripted-v0",
        "lerobot_sim_lab.envs.so100_scripted_env:SO100PickCubeScriptedEnv",
    )
    _register_once(
        "gym_hil/SO100GrabPenBase-v0",
        "lerobot_sim_lab.envs.so100_gym_env:SO100GrabPenGymEnv",
    )


__all__ = ["register_envs"]
