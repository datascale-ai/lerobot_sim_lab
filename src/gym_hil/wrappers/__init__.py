"""Compatibility exports for wrappers."""

from lerobot_sim_lab.envs.wrappers.factory import make_env, wrap_env
from lerobot_sim_lab.envs.wrappers.hil_wrappers import (
    DEFAULT_EE_STEP_SIZE,
    EEActionWrapper,
    GripperPenaltyWrapper,
    InputsControlWrapper,
)
from lerobot_sim_lab.envs.wrappers.viewer_wrapper import PassiveViewerWrapper

__all__ = [
    "DEFAULT_EE_STEP_SIZE",
    "EEActionWrapper",
    "GripperPenaltyWrapper",
    "InputsControlWrapper",
    "PassiveViewerWrapper",
    "make_env",
    "wrap_env",
]
