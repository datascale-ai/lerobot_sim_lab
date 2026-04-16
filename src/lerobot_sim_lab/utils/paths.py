"""Filesystem helpers for package assets and outputs."""

import os
from pathlib import Path

from lerobot_sim_lab.config.defaults import ASSETS_DIR, OUTPUTS_DIR, SO100_MODELS_DIR, SO101_MODELS_DIR


def ensure_dir(path: Path) -> Path:
    """Create a directory if it does not exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_assets_dir() -> Path:
    """Resolve the assets directory, honoring environment overrides."""
    override = os.getenv("LEROBOT_SIM_LAB_ASSETS")
    if override:
        return Path(override).expanduser().resolve()
    return ASSETS_DIR


def get_assets_dir() -> Path:
    """Return the resolved assets directory."""
    return resolve_assets_dir()


def get_so100_models_dir() -> Path:
    """Return the SO-100 model directory."""
    override_assets = resolve_assets_dir()
    if override_assets != ASSETS_DIR:
        return override_assets / "robots" / "so100" / "so100_6dof"
    return SO100_MODELS_DIR


def get_so100_model_path(filename: str) -> Path:
    """Return a specific file path from the SO-100 model directory."""
    return get_so100_models_dir() / filename


def get_so100_scene_path(scene: str = "scene") -> Path:
    """Resolve a named SO-100 scene XML file."""
    scene_map = {
        "basic": "so100.xml",
        "initial": "so100_initial.xml",
        "push_cube": "push_cube_loop.xml",
        "scene": "scene.xml",
    }
    return get_so100_model_path(scene_map.get(scene, scene))


def get_so100_urdf_path() -> Path:
    """Return the default SO-100 URDF path."""
    return get_so100_model_path("so100.urdf")


def get_so100_srdf_path() -> Path:
    """Return the default SO-100 SRDF path."""
    return get_so100_model_path("so100_mplib.srdf")


def get_so101_models_dir() -> Path:
    """Return the SO-101 model directory."""
    override_assets = resolve_assets_dir()
    if override_assets != ASSETS_DIR:
        return override_assets / "robots" / "so101"
    return SO101_MODELS_DIR


def get_outputs_dir() -> Path:
    """Return the default outputs directory."""
    return ensure_dir(OUTPUTS_DIR)


def resolve_output_path(*parts: str) -> Path:
    """Resolve a path inside the default outputs directory."""
    return get_outputs_dir().joinpath(*parts)
