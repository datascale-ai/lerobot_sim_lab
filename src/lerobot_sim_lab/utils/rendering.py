"""Rendering helpers shared by scripts and modules."""

from pathlib import Path

import mediapy as media
import numpy as np

from lerobot_sim_lab.utils.paths import resolve_output_path


def save_video(frames: list[np.ndarray], filename: str, fps: int = 30) -> Path:
    """Save an RGB frame sequence to the outputs directory."""
    output_path = resolve_output_path(filename)
    media.write_video(output_path.as_posix(), frames, fps=fps)
    return output_path
