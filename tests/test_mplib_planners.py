"""Tests for MPlib planners — guarded by optional dependency."""

import numpy as np
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


class TestPlanners:
    def test_plan_qpos_basic(self, planner):
        start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        goal = np.array([0.5, 0.3, -0.2, 0.1, 0.0, 0.0])

        result = planner.plan_qpos(
            goal_qposes=[goal],
            current_qpos=start,
            time_step=0.01,
            planning_time=1.0,
        )
        assert isinstance(result, dict)
        assert "status" in result

    @pytest.mark.parametrize(
        "planner_name",
        ["RRTConnect", "RRTstar", "PRM", "KPIECE", "EST", "BiTRRT", "BiEST"],
    )
    def test_different_planners(self, planner, planner_name):
        start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        goal = np.array([0.5, 0.3, -0.2, 0.1, 0.0, 0.0])

        try:
            result = planner.plan_qpos(
                goal_qposes=[goal],
                current_qpos=start,
                time_step=0.01,
                planning_time=1.0,
                planner_name=planner_name,
                verbose=False,
            )
            assert isinstance(result, dict)
        except TypeError:
            pytest.skip("plan_qpos does not support planner_name parameter")
