"""Tests for MPlib API — guarded by optional dependency."""

import pytest

mplib = pytest.importorskip("mplib")

from lerobot_sim_lab.utils.paths import get_so100_srdf_path, get_so100_urdf_path


@pytest.fixture
def planner():
    """Create an MPlib planner for SO-100."""
    return mplib.Planner(
        urdf=str(get_so100_urdf_path()),
        srdf=str(get_so100_srdf_path()),
        move_group="Moving Jaw",
        verbose=False,
    )


class TestMPLibAPI:
    def test_planner_creation(self, planner):
        assert planner is not None

    def test_planner_has_plan_qpos(self, planner):
        assert hasattr(planner, "plan_qpos")
        assert callable(planner.plan_qpos)

    def test_plan_methods_exist(self, planner):
        methods = [m for m in dir(planner) if "plan" in m.lower() and callable(getattr(planner, m))]
        assert len(methods) > 0
