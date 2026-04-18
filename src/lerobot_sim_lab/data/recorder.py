"""
MuJoCo episode recorder — collect simulation data into HDF5 and convert to LeRobot format.

This is a library module (no CLI entry point). Typical usage from a notebook:

    from lerobot_sim_lab.data.recorder import MuJoCoEpisodeRecorder, LeRobotDatasetConverter
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import mujoco
import numpy as np
from tqdm import tqdm


class MuJoCoEpisodeRecorder:
    """Record MuJoCo simulation episodes into memory, then flush to HDF5."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        fps: int = 30,
        image_size: Tuple[int, int] = (480, 640),
        cameras: Optional[List[str]] = None,
    ):
        self.model = model
        self.data = data
        self.fps = fps
        self.image_size = image_size
        self.cameras = cameras or ["default"]
        self.renderer = mujoco.Renderer(model, height=image_size[0], width=image_size[1])
        self.current_episode: Dict = {}
        self.episodes: List[Dict] = []
        self.start_episode()

    def start_episode(self):
        self.current_episode = {
            "observations": {
                "qpos": [],
                "qvel": [],
                "images": {cam: [] for cam in self.cameras},
            },
            "actions": [],
            "rewards": [],
            "timestamp": [],
        }

    def record_step(self, action: np.ndarray, reward: float = 0.0,
                    camera_configs: Optional[Dict[str, Dict]] = None):
        ep = self.current_episode
        ep["observations"]["qpos"].append(self.data.qpos.copy())
        ep["observations"]["qvel"].append(self.data.qvel.copy())
        ep["actions"].append(action.copy())
        ep["rewards"].append(reward)
        ep["timestamp"].append(self.data.time)

        for cam_name in self.cameras:
            if camera_configs and cam_name in camera_configs:
                cfg = camera_configs[cam_name]
                cam = mujoco.MjvCamera()
                cam.lookat = np.array(cfg.get("lookat", [0, 0.08, 0.15]))
                cam.distance = cfg.get("distance", 0.6)
                cam.azimuth = cfg.get("azimuth", 180)
                cam.elevation = cfg.get("elevation", -10)
                self.renderer.update_scene(self.data, camera=cam)
            else:
                self.renderer.update_scene(self.data)
            ep["observations"]["images"][cam_name].append(self.renderer.render())

    def end_episode(self):
        if not self.current_episode["actions"]:
            return
        episode_data = {
            "observations": {
                "qpos": np.array(self.current_episode["observations"]["qpos"]),
                "qvel": np.array(self.current_episode["observations"]["qvel"]),
                "images": {
                    cam: np.array(self.current_episode["observations"]["images"][cam])
                    for cam in self.cameras
                },
            },
            "actions": np.array(self.current_episode["actions"]),
            "rewards": np.array(self.current_episode["rewards"]),
            "timestamp": np.array(self.current_episode["timestamp"]),
            "length": len(self.current_episode["actions"]),
        }
        self.episodes.append(episode_data)

    def save_raw_hdf5(self, output_path: str):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_path, "w") as f:
            meta = f.create_group("metadata")
            meta.attrs["fps"] = self.fps
            meta.attrs["image_height"] = self.image_size[0]
            meta.attrs["image_width"] = self.image_size[1]
            meta.attrs["num_episodes"] = len(self.episodes)
            meta.attrs["total_frames"] = sum(ep["length"] for ep in self.episodes)
            meta.attrs["cameras"] = json.dumps(self.cameras)
            meta.attrs["created_at"] = datetime.now().isoformat()
            data_grp = f.create_group("data")
            for idx, ep in enumerate(tqdm(self.episodes, desc="Saving episodes")):
                eg = data_grp.create_group(f"episode_{idx}")
                obs = eg.create_group("observations")
                obs.create_dataset("qpos", data=ep["observations"]["qpos"], compression="gzip")
                obs.create_dataset("qvel", data=ep["observations"]["qvel"], compression="gzip")
                img_grp = obs.create_group("images")
                for cam, imgs in ep["observations"]["images"].items():
                    img_grp.create_dataset(cam, data=imgs, compression="gzip")
                eg.create_dataset("actions", data=ep["actions"], compression="gzip")
                eg.create_dataset("rewards", data=ep["rewards"], compression="gzip")
                eg.create_dataset("timestamp", data=ep["timestamp"], compression="gzip")
                eg.attrs["length"] = ep["length"]

    def get_stats(self) -> Dict:
        if not self.episodes:
            return {"num_episodes": 0, "total_frames": 0}
        return {
            "num_episodes": len(self.episodes),
            "total_frames": sum(ep["length"] for ep in self.episodes),
            "avg_episode_length": float(np.mean([ep["length"] for ep in self.episodes])),
            "action_shape": self.episodes[0]["actions"].shape[1:],
            "cameras": self.cameras,
        }


class LeRobotDatasetConverter:
    """Convert raw HDF5 data to LeRobot dataset format."""

    @staticmethod
    def convert(hdf5_path: str, output_dir: str, repo_id: str = "local/mujoco_sim",
                robot_type: str = "so100", fps: int = 30):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        with h5py.File(hdf5_path, "r") as f:
            metadata = dict(f["metadata"].attrs)
            num_episodes = metadata["num_episodes"]
            cameras = json.loads(metadata["cameras"])
            ep0 = f["data/episode_0"]
            qpos_dim = ep0["observations/qpos"].shape[1]
            qvel_dim = ep0["observations/qvel"].shape[1]
            action_dim = ep0["actions"].shape[1]
            img_shape = ep0[f"observations/images/{cameras[0]}"].shape[1:]

            features = {
                "observation.state": {"dtype": "float32", "shape": (qpos_dim + qvel_dim,), "names": ["qpos_qvel"]},
                "action": {"dtype": "float32", "shape": (action_dim,), "names": ["joint_positions"]},
            }
            for cam in cameras:
                features[f"observation.images.{cam}"] = {
                    "dtype": "video",
                    "shape": (img_shape[2], img_shape[0], img_shape[1]),
                    "names": ["channel", "height", "width"],
                }

            dataset = LeRobotDataset.create(
                repo_id=repo_id, fps=fps, root=output_dir,
                robot_type=robot_type, features=features, use_videos=True,
            )
            dataset.start_image_writer(num_processes=0, num_threads=4)

            for ep_idx in tqdm(range(num_episodes), desc="Converting"):
                eg = f[f"data/episode_{ep_idx}"]
                qpos = eg["observations/qpos"][:]
                qvel = eg["observations/qvel"][:]
                actions = eg["actions"][:]
                for i in range(len(actions)):
                    frame = {
                        "observation.state": np.concatenate([qpos[i], qvel[i]]).astype(np.float32),
                        "action": actions[i].astype(np.float32),
                    }
                    for cam in cameras:
                        img = eg[f"observations/images/{cam}"][i]
                        frame[f"observation.images.{cam}"] = np.transpose(img, (2, 0, 1))
                    dataset.add_frame(frame)
                dataset.save_episode()

            dataset.stop_image_writer()
            dataset.consolidate()
            return dataset
