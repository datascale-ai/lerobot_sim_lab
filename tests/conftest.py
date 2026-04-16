"""Shared pytest fixtures for lerobot_sim_lab."""

from pathlib import Path

import pytest

from lerobot_sim_lab.utils.paths import get_so100_models_dir


@pytest.fixture
def mujoco_scene_xml() -> Path:
    """Return the default SO-100 scene XML path."""
    return get_so100_models_dir() / "scene.xml"


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Provide a temporary output directory."""
    out = tmp_path / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out
