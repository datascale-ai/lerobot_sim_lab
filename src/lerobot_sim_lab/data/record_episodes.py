#!/usr/bin/env python3
"""
Record scripted SO100 episodes (high-performance rendering).

Modes:
  scripted    — run the scripted gym environment and record episodes
  trajectory  — replay pre-generated trajectories and record episodes

CLI entry point: lerobot-sim-record
"""

import argparse
import logging
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch  # noqa: F401 — needed by lerobot at import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _lazy_lerobot_imports():
    """Import lerobot components lazily so the module loads without lerobot installed."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import ACTION, DONE, REWARD
    return LeRobotDataset, ACTION, DONE, REWARD


def _lazy_scenario_imports():
    """Import scenario helpers."""
    from lerobot_sim_lab.config.scenarios.pick_place import get_num_scenarios, get_scenario
    return get_num_scenarios, get_scenario


def _lazy_pen_imports():
    from lerobot_sim_lab.config.scenarios.pen_grab import PEN_SCENARIOS, PEN_QPOS_MAP, BOX_POSITION, BOX_QUATERNION, BOX_QPOS_START
    return PEN_SCENARIOS, PEN_QPOS_MAP, BOX_POSITION, BOX_QUATERNION, BOX_QPOS_START


def create_dataset(repo_id: str, root: str, fps: int):
    """Create a standard LeRobotDataset for recording."""
    LeRobotDataset, ACTION, DONE, REWARD = _lazy_lerobot_imports()
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": [f"robot_qpos_{i}" for i in range(6)],
        },
        "observation.images.front": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        },
        ACTION: {
            "dtype": "float32",
            "shape": (6,),
            "names": [f"joint_{i}" for i in range(6)],
        },
        REWARD: {"dtype": "float32", "shape": (1,), "names": None},
        DONE: {"dtype": "bool", "shape": (1,), "names": None},
    }
    dataset = LeRobotDataset.create(
        repo_id=repo_id, fps=fps, root=root,
        robot_type="so100", features=features, use_videos=True,
    )
    dataset.start_image_writer(num_processes=0, num_threads=4)
    return dataset


def render_dual_view(env, renderer):
    """Render front + side camera views using XML-defined cameras."""
    import mujoco
    model = env.unwrapped._model
    data = env.unwrapped._data

    def _render_camera(cam_name):
        try:
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
            renderer.update_scene(data, camera=cam_id)
            return renderer.render()
        except Exception:
            return np.zeros((480, 640, 3), dtype=np.uint8)

    frame_front = _render_camera("camera_front_new")
    frame_side = _render_camera("camera_side")
    return frame_front, frame_side


def render_dual_view_mujoco(model, data, renderer):
    """Render front + wrist views from raw model/data (no gym env)."""
    import mujoco

    def _render_camera(cam_name):
        try:
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
            renderer.update_scene(data, camera=cam_id)
            return renderer.render()
        except Exception:
            return np.zeros((480, 640, 3), dtype=np.uint8)

    return _render_camera("camera_front_new"), _render_camera("camera_wrist")


def set_pens_positions(model, data, pens_config):
    import mujoco
    _, PEN_QPOS_MAP, BOX_POSITION, BOX_QUATERNION, BOX_QPOS_START = _lazy_pen_imports()
    data.qpos[BOX_QPOS_START:BOX_QPOS_START + 3] = BOX_POSITION
    data.qpos[BOX_QPOS_START + 3:BOX_QPOS_START + 7] = BOX_QUATERNION
    for pen_name, (pos, quat) in pens_config.items():
        if pen_name in PEN_QPOS_MAP:
            qpos_start = PEN_QPOS_MAP[pen_name]
            data.qpos[qpos_start:qpos_start + 3] = pos
            data.qpos[qpos_start + 3:qpos_start + 7] = quat
    mujoco.mj_forward(model, data)


def load_seed_indices(seed_file: Path):
    if not seed_file.exists():
        return []
    content = seed_file.read_text().strip()
    if not content:
        return []
    return [int(x) for x in content.split(",") if x.strip() != ""]


def get_pen_scenario(scenario_id: int):
    PEN_SCENARIOS = _lazy_pen_imports()[0]
    for scenario in PEN_SCENARIOS:
        if scenario["id"] == scenario_id:
            return scenario
    return None


def record_episode(env, dataset, episode_idx, scenario_idx=1, task="pick_cube", fps=30):
    """Record a single scripted episode."""
    import mujoco
    _, ACTION, DONE, REWARD = _lazy_lerobot_imports()
    get_num_scenarios, get_scenario = _lazy_scenario_imports()

    try:
        scenario = get_scenario(scenario_idx)
        logger.info(f"Recording episode {episode_idx} | scenario {scenario_idx}: {scenario['name']}")
    except Exception:
        scenario_idx = 1
        scenario = get_scenario(scenario_idx)
        logger.info(f"Fallback to scenario {scenario_idx}: {scenario['name']}")

    start_time = time.time()
    model = env.unwrapped._model
    renderer = mujoco.Renderer(model, height=480, width=640)

    obs, info = env.reset(options={"scenario_idx": scenario_idx})
    terminated = truncated = False
    step_count = 0
    total_reward = 0
    frames_buffer = []

    while not (terminated or truncated):
        action = env.unwrapped.get_scripted_action()
        frame_front, frame_side = render_dual_view(env, renderer)
        next_obs, reward, terminated, truncated, info = env.step(action)

        if frame_front is not None:
            frames_buffer.append({
                "observation.state": obs["state"][:6].copy(),
                "observation.images.front": frame_front.copy(),
                "observation.images.wrist": frame_side.copy(),
                ACTION: action.astype(np.float32).copy(),
                REWARD: np.array([reward], dtype=np.float32),
                DONE: np.array([terminated or truncated], dtype=bool),
                "task": task,
            })
        obs = next_obs
        step_count += 1
        total_reward += reward

    sim_time = time.time() - start_time
    logger.info(f"  Sim done in {sim_time:.2f}s ({step_count/sim_time:.1f} steps/s), {len(frames_buffer)} frames")

    for frame in frames_buffer:
        dataset.add_frame(frame)
    dataset.save_episode()

    success = info.get("succeed", False)
    return success, step_count, total_reward


def record_trajectory_episode(model, data, renderer, dataset, trajectory,
                              episode_idx, scenario, task, fps):
    """Record a single trajectory-replay episode."""
    import mujoco
    _, ACTION, DONE, REWARD = _lazy_lerobot_imports()

    start_time = time.time()
    home_keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_keyframe_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_keyframe_id)
    else:
        mujoco.mj_resetData(model, data)

    set_pens_positions(model, data, scenario["pens"])
    data.ctrl[:6] = trajectory[0]
    data.qpos[:6] = trajectory[0]
    mujoco.mj_forward(model, data)

    frames_buffer = []
    n_substeps = max(1, int((1 / fps) / model.opt.timestep))

    for qpos in trajectory:
        data.ctrl[:6] = qpos
        for _ in range(n_substeps):
            mujoco.mj_step(model, data)
        frame_front, frame_wrist = render_dual_view_mujoco(model, data, renderer)
        frames_buffer.append({
            "observation.state": data.qpos[:6].copy().astype(np.float32),
            "observation.images.front": frame_front.copy(),
            "observation.images.wrist": frame_wrist.copy(),
            ACTION: qpos.astype(np.float32).copy(),
            REWARD: np.array([0.0], dtype=np.float32),
            DONE: np.array([False], dtype=bool),
            "task": task,
        })

    if frames_buffer:
        frames_buffer[-1][DONE] = np.array([True], dtype=bool)

    sim_time = time.time() - start_time
    logger.info(f"  Episode {episode_idx} done in {sim_time:.2f}s, {len(frames_buffer)} frames")

    for frame in frames_buffer:
        dataset.add_frame(frame)
    dataset.save_episode()
    return len(frames_buffer)


def main():
    parser = argparse.ArgumentParser(description="Record scripted SO100 episodes")
    parser.add_argument("--mode", type=str, default="scripted", choices=["scripted", "trajectory"])
    parser.add_argument("--env-id", type=str, default="lerobot_sim_lab/SO100PickCubeScripted-v0")
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--repo-id", type=str, default="so100_pick_scripted")
    parser.add_argument("--root", type=str, default="./data")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--task", type=str,
                        default="Pick up the red cube and place it in the target region")
    parser.add_argument("--trajectory-root", type=str, default="pen_grab_tuning")
    parser.add_argument("--scenarios", type=str, default="1,2,3,4,5")
    parser.add_argument("--seed-file", type=str, default="seed.txt")
    args = parser.parse_args()

    import lerobot_sim_lab.envs  # noqa: F401 — register envs
    lerobot_sim_lab.envs.register_envs()

    dataset = create_dataset(repo_id=args.repo_id, root=args.root, fps=args.fps)

    if args.mode == "scripted":
        _run_scripted(args, dataset)
    else:
        _run_trajectory(args, dataset)


def _run_scripted(args, dataset):
    get_num_scenarios, get_scenario = _lazy_scenario_imports()
    num_scenarios = get_num_scenarios()
    logger.info(f"Scenario library: {num_scenarios} verified scenarios")

    env = gym.make(args.env_id, image_obs=False, control_dt=1 / 30)
    success_count = 0

    for episode_idx in range(args.num_episodes):
        scenario_idx = (episode_idx % num_scenarios) + 1
        success, _, _ = record_episode(
            env, dataset, episode_idx,
            scenario_idx=scenario_idx, task=args.task, fps=args.fps,
        )
        if success:
            success_count += 1

    logger.info(f"Done: {success_count}/{args.num_episodes} successful")
    dataset.stop_image_writer()
    env.close()


def _run_trajectory(args, dataset):
    import mujoco
    from lerobot_sim_lab.utils.paths import get_so100_models_dir

    model_path = get_so100_models_dir() / "scene.xml"
    model = mujoco.MjModel.from_xml_path(model_path.as_posix())
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)

    scenario_ids = [int(x) for x in args.scenarios.split(",") if x.strip()]
    episode_idx = 0
    for scenario_id in scenario_ids:
        scenario = get_pen_scenario(scenario_id)
        if scenario is None:
            logger.warning(f"Scenario {scenario_id} not found, skip")
            continue
        scenario_dir = Path(args.trajectory_root) / f"scenario_{scenario_id}"
        traj_dir = scenario_dir / "trajectories"
        seed_file = scenario_dir / args.seed_file
        seeds = load_seed_indices(seed_file)
        if not seeds:
            continue
        if len(seeds) > 12:
            seeds = list(np.random.default_rng(42).choice(seeds, size=12, replace=False))
        for seed in seeds:
            traj_file = traj_dir / f"episode_{seed:03d}.npz"
            if not traj_file.exists():
                continue
            trajectory = np.load(traj_file)["trajectory"]
            logger.info(f"Episode {episode_idx} | scenario {scenario_id} | seed {seed}")
            record_trajectory_episode(
                model, data, renderer, dataset, trajectory,
                episode_idx, scenario, args.task, args.fps,
            )
            episode_idx += 1

    dataset.stop_image_writer()
    renderer.close()


if __name__ == "__main__":
    main()
