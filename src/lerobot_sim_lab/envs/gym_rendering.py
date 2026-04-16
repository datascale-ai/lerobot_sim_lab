"""Rendering dataclasses for MuJoCo gym environments."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class GymRenderingSpec:
    """Renderer configuration for offscreen MuJoCo rendering."""

    height: int = 128
    width: int = 128
    camera_id: str | int = -1
    mode: Literal["rgb_array", "human"] = "rgb_array"
