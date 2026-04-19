"""
Dataset visualization utilities for HDF5 datasets.

Library module — designed for use in Jupyter notebooks:

    from lerobot_sim_lab.data.visualize import quick_view, full_analysis
"""

import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
from matplotlib.gridspec import GridSpec


def load_dataset_info(hdf5_path: str):
    """Print and return basic dataset metadata."""
    with h5py.File(hdf5_path, "r") as f:
        meta = dict(f["metadata"].attrs)
        num_episodes = meta["num_episodes"]
        cameras = json.loads(meta["cameras"])

        print("=" * 70)
        print(f"Dataset: {Path(hdf5_path).name}")
        print("=" * 70)
        print(f"  Episodes: {num_episodes}")
        print(f"  FPS: {meta['fps']}")
        print(f"  Image size: {meta['image_height']} x {meta['image_width']}")
        print(f"  Total frames: {meta['total_frames']}")
        print(f"  Cameras: {cameras}")

        lengths = [f[f"data/episode_{i}"].attrs["length"] for i in range(num_episodes)]
        print(f"  Avg length: {np.mean(lengths):.1f}  min: {np.min(lengths)}  max: {np.max(lengths)}")
        return meta, lengths


def visualize_episode(hdf5_path: str, episode_idx: int = 0, save_video: bool = True, output_dir: str = "."):
    """Visualize a single episode: joint curves + video."""
    with h5py.File(hdf5_path, "r") as f:
        meta = dict(f["metadata"].attrs)
        cameras = json.loads(meta["cameras"])
        eg = f[f"data/episode_{episode_idx}"]

        qpos = eg["observations/qpos"][:]
        actions = eg["actions"][:]
        rewards = eg["rewards"][:]

        fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        for i in range(min(6, qpos.shape[1])):
            ax1.plot(qpos[:, i], label=f"J{i}", alpha=0.7)
        ax1.set_title("Joint positions")
        ax1.legend(ncol=3, fontsize=7)
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[0, 1])
        for i in range(min(6, actions.shape[1])):
            ax2.plot(actions[:, i], label=f"A{i}", alpha=0.7)
        ax2.set_title("Actions")
        ax2.legend(ncol=3, fontsize=7)
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(rewards, color="green")
        ax3.set_title(f"Reward (mean {np.mean(rewards):.3f})")
        ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(gs[1, 1])
        first_frames = [eg[f"observations/images/{c}"][0] for c in cameras]
        ax4.imshow(np.concatenate(first_frames, axis=1))
        ax4.set_title(f"Frame 0 ({' + '.join(cameras)})")
        ax4.axis("off")

        plt.tight_layout()
        plot_path = Path(output_dir) / f"episode_{episode_idx}_analysis.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.show()

        if save_video:
            video_path = Path(output_dir) / f"episode_{episode_idx}_video.mp4"
            frames = [
                np.concatenate([eg[f"observations/images/{c}"][i] for c in cameras], axis=1)
                for i in range(len(actions))
            ]
            media.write_video(str(video_path), frames, fps=meta["fps"])
            print(f"Video saved: {video_path}")


def quick_view(hdf5_path: str, episode_idx: int = 0):
    load_dataset_info(hdf5_path)
    visualize_episode(hdf5_path, episode_idx)


def full_analysis(hdf5_path: str):
    meta, _ = load_dataset_info(hdf5_path)
    visualize_episode(hdf5_path, 0)
