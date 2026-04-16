"""Default paths and constants for lerobot_sim_lab."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
ASSETS_DIR = PROJECT_ROOT / "assets"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SO100_MODELS_DIR = ASSETS_DIR / "robots" / "so100" / "so100_6dof"
SO101_MODELS_DIR = ASSETS_DIR / "robots" / "so101"
JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
CONTROL_STEP = 0.05
GRIPPER_STEP = 0.1
