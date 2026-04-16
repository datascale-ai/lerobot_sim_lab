"""SO-100 MuJoCo environments."""

from typing import Any, Literal

import mujoco
import numpy as np
from gymnasium import spaces

from lerobot_sim_lab.config.scenarios.pen_grab import (
    BOX_POSITION,
    BOX_QPOS_START,
    BOX_QUATERNION,
    PEN_QPOS_MAP,
    PEN_SCENARIOS,
)
from lerobot_sim_lab.envs.gym_rendering import GymRenderingSpec
from lerobot_sim_lab.envs.mujoco_gym_env import MujocoGymEnv
from lerobot_sim_lab.utils.paths import get_so100_models_dir

# 场景边界
_SAMPLING_BOUNDS = np.array(
    [[0.025, 0.09], [-0.095, 0.18]]  # X 范围（黄色区域左边界，工作区底部）  # Y 范围（紫色区域左边界，工作区顶部）
)

# Cube 目标位置
_GOAL_POSITIONS = {
    "yellow": np.array([0.06, 0.135]),  # 黄色区域中心
    "purple": np.array([-0.06, 0.135]),  # 紫色区域中心
}


class SO100PickCubeGymEnv(MujocoGymEnv):
    """
    SO-100 机械臂抓取cube任务环境

    任务：将cube从黄色区域抓取并放置到紫色区域

    观测空间：
        - state: 机器人关节位置、速度、cube位置等
        - images: 前置相机、侧面相机

    动作空间：
        - 4D: (delta_x, delta_y, delta_z, gripper)
        - 末端执行器的增量位置 + 夹爪控制

    奖励：
        - sparse: 成功放置 cube 到目标区域得 1.0
        - dense: 基于距离的连续奖励
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        render_mode: Literal["rgb_array", "human"] | None = None,
        observation_width: int = 640,
        observation_height: int = 480,
        image_obs: bool = True,
        reward_type: str = "sparse",
        success_threshold: float = 0.05,  # 成功距离阈值（米）
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        use_gripper: bool = True,  # gym_manipulator 传递的参数
        gripper_penalty: float = 0.0,  # gym_manipulator 传递的参数
        **kwargs,  # 接受其他可能的参数
    ):
        # print(f"DEBUG: SO100PickCubeGymEnv init with image_obs={image_obs}, kwargs={kwargs}")
        """
        初始化SO-100抓取环境

        Args:
            render_mode: 渲染模式 ("human" 或 "rgb_array")
            observation_width: 观测图像宽度
            observation_height: 观测图像高度
            image_obs: 是否包含图像观测
            reward_type: 奖励类型 ("sparse" 或 "dense")
            success_threshold: 成功判定的距离阈值
            seed: 随机种子
            control_dt: 控制时间步长
            physics_dt: 物理仿真时间步长
        """
        # SO-100 场景 XML 文件路径
        # 使用与 demo_pick_and_place.py 相同的场景
        xml_path = get_so100_models_dir() / "push_cube_loop.xml"

        # 渲染配置
        render_spec = GymRenderingSpec(
            height=observation_height,
            width=observation_width,
            camera_id="camera_front_new",
        )

        super().__init__(
            xml_path=xml_path,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            render_spec=render_spec,
        )
        self.render_mode = render_mode
        self.image_obs = image_obs
        self.reward_type = reward_type
        self.success_threshold = success_threshold
        self.task = "lerobot_sim_lab/SO100PickCube-v0"
        self.task_description = "Pick up the cube from the table and place it into the target region."

        # SO-100 关节名称（与 XML 文件中的定义一致）
        # 有效的关节: ['Rotation', 'Pitch', 'Elbow', 'Wrist_Pitch', 'Wrist_Roll', 'Jaw']
        self._joint_names = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]

        # 缓存关节和执行器 ID
        self._so100_joint_ids = []
        self._so100_ctrl_ids = []
        for joint_name in self._joint_names:
            try:
                joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                self._so100_joint_ids.append(joint_id)
                # 假设执行器名称与关节名称相同或有规律
                try:
                    ctrl_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
                    self._so100_ctrl_ids.append(ctrl_id)
                except Exception:
                    # 如果找不到同名执行器，使用索引
                    self._so100_ctrl_ids.append(len(self._so100_ctrl_ids))
            except KeyError:
                print(f"Warning: Joint {joint_name} not found in model")

        self._so100_joint_ids = np.array(self._so100_joint_ids)
        self._so100_ctrl_ids = np.array(self._so100_ctrl_ids)

        # Cube 初始位置（黄色区域）
        self._initial_cube_pos = np.array([0.06, 0.135, 0.017])

        # 目标位置（紫色区域）
        self._target_pos = np.array([-0.06, 0.135, 0.017])

        # 机器人初始关节配置
        # 使用 demo_pick_and_place.py 中的初始姿态
        self._initial_qpos = np.array([0, -3.14, 3.14, 0.817, 0, -0.157])

        # 设置观测和动作空间
        self._setup_spaces()

        # 设置元数据
        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": int(np.round(1.0 / control_dt)),
        }

        # 计算子步数
        self._n_substeps = int(control_dt / physics_dt)

    def _setup_spaces(self):
        """设置观测空间和动作空间"""
        # 动作空间：6个关节的位置控制
        self.action_space = spaces.Box(
            low=-np.pi,
            high=np.pi,
            shape=(6,),
            dtype=np.float32,
        )

        # 观测空间
        # state: robot_qpos (6)
        state_dim = 6

        obs_spaces = {
            "state": spaces.Box(-np.inf, np.inf, shape=(state_dim,), dtype=np.float32),
        }

        if self.image_obs:
            obs_spaces["pixels"] = spaces.Dict(
                {
                    "front": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self._render_specs.height, self._render_specs.width, 3),
                        dtype=np.uint8,
                    ),
                    # Hack: Add side camera (duplicate of front) to match pretrained model config
                    "side": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self._render_specs.height, self._render_specs.width, 3),
                        dtype=np.uint8,
                    ),
                }
            )

        self.observation_space = spaces.Dict(obs_spaces)

    def reset(self, seed=None, options=None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """重置环境

        Args:
            seed: Random seed
            options: Optional dict with reset options. Can include 'cube_pos' key.
        """
        # Ensure gymnasium internal RNG is initialized when a seed is provided
        super().reset(seed=seed)

        mujoco.mj_resetData(self._model, self._data)

        # 重置机器人到初始位置
        self.reset_robot()

        # 设置cube位置（支持自定义位置）
        if options and "cube_pos" in options:
            cube_x, cube_y, cube_z = options["cube_pos"]
        else:
            # 使用默认位置
            cube_x = self._initial_cube_pos[0]  # 固定在 0.06
            cube_y = self._initial_cube_pos[1]  # 固定在 0.135
            cube_z = self._initial_cube_pos[2]  # 固定在 0.017

        # 设置cube位置
        # 假设cube是自由体，qpos索引从6开始
        if len(self._data.qpos) >= 10:  # 6 (robot) + 3 (cube pos) + 4 (cube quat)
            self._data.qpos[6:9] = [cube_x, cube_y, cube_z]
            self._data.qpos[9:13] = [1, 0, 0, 0]  # 单位四元数

        mujoco.mj_forward(self._model, self._data)
        obs = self._compute_observation()

        return obs, {}

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """执行一步动作"""
        # 应用动作到机器人（直接设置关节位置）
        if len(action) == 6:
            self._data.ctrl[:6] = action

        # 执行仿真步骤
        for _ in range(self._n_substeps):
            mujoco.mj_step(self._model, self._data)

        # 计算观测、奖励和终止条件
        obs = self._compute_observation()
        reward = self._compute_reward()
        success = self._is_success()

        if self.reward_type == "sparse":
            reward = 1.0 if success else 0.0

        # 检查cube是否超出边界（使用更宽松的工作空间边界）
        cube_pos = self._get_cube_position()
        # 整个工作台的边界（根据 XML 中的 walls 定义）
        workspace_x = [-0.125, 0.125]  # left_wall to right_wall
        workspace_y = [0.09, 0.18]  # top_wall to bottom_wall

        out_of_bounds = (
            cube_pos[0] < workspace_x[0] - 0.02
            or cube_pos[0] > workspace_x[1] + 0.02
            or cube_pos[1] < workspace_y[0] - 0.02
            or cube_pos[1] > workspace_y[1] + 0.02
        )

        # Cube掉落到地面
        cube_dropped = cube_pos[2] < 0.005

        # 完全禁用提前终止，让机械臂执行完整个动作序列
        # 依赖 TimeLimit wrapper 来控制 episode 长度
        terminated = False

        info = {
            "is_success": success,
            "out_of_bounds": out_of_bounds,
            "cube_dropped": cube_dropped,
        }

        return obs, reward, terminated, False, info

    def _compute_observation(self) -> dict:
        """计算当前观测"""
        obs = {}
        # DEBUG PRINT
        # print(f"DEBUG: Computing observation, image_obs={self.image_obs}")

        # 机器人状态
        robot_qpos = self._data.qpos[:6].copy()

        # Cube 状态
        cube_pos = self._get_cube_position()
        cube_vel = self._get_cube_velocity() if len(self._data.qvel) >= 12 else np.zeros(3)

        # 目标位置
        target_pos = self._target_pos.copy()

        # 组合状态向量 (仅包含本体感知信息)
        state = robot_qpos

        obs["state"] = state.astype(np.float32)

        # 添加图像（如果启用）
        if self.image_obs:
            # 确保 viewer 已初始化
            if self._viewer is None:
                self._viewer = mujoco.Renderer(
                    model=self._model,
                    height=self._render_specs.height,
                    width=self._render_specs.width,
                )

            # 渲染 Front (camera_front_new)
            try:
                cam_id_front = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "camera_front_new")
                self._viewer.update_scene(self._data, camera=cam_id_front)
                img_front = self._viewer.render()
            except Exception:
                # print(f"Warning: Failed to render front camera: {e}")
                img_front = np.zeros((self._render_specs.height, self._render_specs.width, 3), dtype=np.uint8)

            # 渲染 Side (camera_side)
            try:
                cam_id_side = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "camera_side")
                self._viewer.update_scene(self._data, camera=cam_id_side)
                img_side = self._viewer.render()
            except Exception:
                # print(f"Warning: Failed to render side camera: {e}")
                img_side = np.zeros((self._render_specs.height, self._render_specs.width, 3), dtype=np.uint8)

            obs["pixels"] = {"front": img_front, "side": img_side}

        # print(f"DEBUG: Obs keys: {list(obs.keys())}")
        return obs

    def _compute_reward(self) -> float:
        """计算奖励"""
        cube_pos = self._get_cube_position()
        target_pos = self._target_pos

        # 到目标的距离
        distance_to_target = np.linalg.norm(cube_pos[:2] - target_pos[:2])

        if self.reward_type == "sparse":
            # 稀疏奖励：只在成功时给1
            return 1.0 if distance_to_target < self.success_threshold else 0.0
        else:
            # 密集奖励：基于距离
            reward = -distance_to_target

            # 如果cube被抬起，给额外奖励
            if cube_pos[2] > 0.05:
                reward += 0.1

            # 如果接近目标，给额外奖励
            if distance_to_target < 0.1:
                reward += 0.5

            # 如果成功，给大奖励
            if distance_to_target < self.success_threshold:
                reward += 5.0

            return reward

    def _is_success(self) -> bool:
        """判断是否成功 - cube 在紫色目标区域内"""
        cube_pos = self._get_cube_position()

        # 紫色区域边界（根据 XML 定义）
        # <geom name="goal_region_2" type="box" pos="-.06 .135 0.01" size="0.035 0.045 0.007" />
        purple_region_x = [-0.095, -0.025]  # center -0.06 ± 0.035
        purple_region_y = [0.09, 0.18]  # center 0.135 ± 0.045
        purple_region_z = [0.003, 0.017]  # center 0.01 ± 0.007

        # cube 的 x, y, z 都在紫色区域内
        in_x = purple_region_x[0] <= cube_pos[0] <= purple_region_x[1]
        in_y = purple_region_y[0] <= cube_pos[1] <= purple_region_y[1]
        in_z = purple_region_z[0] <= cube_pos[2] <= purple_region_z[1]

        return in_x and in_y and in_z

    def _get_cube_position(self) -> np.ndarray:
        """获取cube位置"""
        # 尝试从传感器读取
        try:
            return self._data.sensor("cube_pos").data.copy()
        except Exception:
            # 如果没有传感器，从qpos读取
            if len(self._data.qpos) >= 9:
                return self._data.qpos[6:9].copy()
            else:
                return self._initial_cube_pos.copy()

    def _get_cube_velocity(self) -> np.ndarray:
        """获取cube速度"""
        if len(self._data.qvel) >= 12:
            return self._data.qvel[6:9].copy()
        return np.zeros(3)

    def reset_robot(self):
        """重置机器人到初始位置"""
        self._data.qpos[:6] = self._initial_qpos
        self._data.qvel[:6] = 0
        self._data.ctrl[:6] = self._initial_qpos
        mujoco.mj_forward(self._model, self._data)

    def apply_action(self, action: np.ndarray):
        """应用动作到机器人（兼容接口）"""
        if len(action) == 6:
            self._data.ctrl[:6] = action


BOX_BOUNDS = {
    "x": (-0.06, 0.06),
    "y": (-0.08, 0.08),
    "z": (0.0, 0.08),
}
BOX_MARGIN = 0.008


class SO100GrabPenGymEnv(MujocoGymEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        render_mode: Literal["rgb_array", "human"] | None = None,
        observation_width: int = 640,
        observation_height: int = 480,
        image_obs: bool = True,
        reward_type: str = "sparse",
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        use_gripper: bool = True,
        gripper_penalty: float = 0.0,
        scenario_id: int = 1,
        **kwargs,
    ):
        xml_path = get_so100_models_dir() / "scene.xml"
        render_spec = GymRenderingSpec(
            height=observation_height,
            width=observation_width,
            camera_id="camera_front_new",
        )
        super().__init__(
            xml_path=xml_path,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            render_spec=render_spec,
        )
        self.render_mode = render_mode
        self.image_obs = image_obs
        self.reward_type = reward_type
        self.task = "lerobot_sim_lab/SO100GrabPen-v0"
        self.task_description = "Pick up the pen from the table and place it into the box."
        self._joint_names = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
        self._so100_joint_ids = []
        self._so100_ctrl_ids = []
        for joint_name in self._joint_names:
            try:
                joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                self._so100_joint_ids.append(joint_id)
                try:
                    ctrl_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
                    self._so100_ctrl_ids.append(ctrl_id)
                except Exception:
                    self._so100_ctrl_ids.append(len(self._so100_ctrl_ids))
            except KeyError:
                print(f"Warning: Joint {joint_name} not found in model")
        self._so100_joint_ids = np.array(self._so100_joint_ids)
        self._so100_ctrl_ids = np.array(self._so100_ctrl_ids)
        self._initial_qpos = np.array([0, -3.14, 3.14, 0.817, 0, -0.157])
        self._default_scenario_id = scenario_id
        self._pen_names = ["pen1", "pen_2", "pen_3", "pen_4"]
        self._setup_spaces()
        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": int(np.round(1.0 / control_dt)),
        }
        self._n_substeps = int(control_dt / physics_dt)

    def _setup_spaces(self):
        self.action_space = spaces.Box(
            low=-np.pi,
            high=np.pi,
            shape=(6,),
            dtype=np.float32,
        )
        state_dim = 6
        obs_spaces = {
            "state": spaces.Box(-np.inf, np.inf, shape=(state_dim,), dtype=np.float32),
        }
        if self.image_obs:
            obs_spaces["pixels"] = spaces.Dict(
                {
                    "front": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self._render_specs.height, self._render_specs.width, 3),
                        dtype=np.uint8,
                    ),
                    "wrist": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self._render_specs.height, self._render_specs.width, 3),
                        dtype=np.uint8,
                    ),
                }
            )
        self.observation_space = spaces.Dict(obs_spaces)

    def reset(self, seed=None, options=None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        home_keyframe_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_keyframe_id >= 0:
            mujoco.mj_resetDataKeyframe(self._model, self._data, home_keyframe_id)
        else:
            mujoco.mj_resetData(self._model, self._data)
        self._data.qpos[:6] = self._initial_qpos
        self._data.qvel[:6] = 0
        self._data.ctrl[:6] = self._initial_qpos
        if options and "scenario_idx" in options:
            scenario_id = options["scenario_idx"]
        else:
            scenario_ids = [scenario["id"] for scenario in PEN_SCENARIOS if scenario["id"] != 0]
            scenario_id = int(self._random.choice(scenario_ids))
        scenario = self._get_pen_scenario(scenario_id)
        if scenario is None:
            scenario = self._get_pen_scenario(self._default_scenario_id)
        if scenario is not None:
            self._set_pens_positions(scenario["pens"])
        mujoco.mj_forward(self._model, self._data)
        obs = self._compute_observation()
        return obs, {"scenario_id": scenario_id}

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if len(action) == 6:
            self._data.ctrl[:6] = action
        for _ in range(self._n_substeps):
            mujoco.mj_step(self._model, self._data)
        obs = self._compute_observation()
        reward = self._compute_reward()
        success = self._is_success()
        info = {"is_success": success}
        terminated = False
        return obs, reward, terminated, False, info

    def _compute_observation(self) -> dict:
        obs = {}
        robot_qpos = self._data.qpos[:6].copy()
        obs["state"] = robot_qpos.astype(np.float32)
        if self.image_obs:
            if self._viewer is None:
                self._viewer = mujoco.Renderer(
                    model=self._model,
                    height=self._render_specs.height,
                    width=self._render_specs.width,
                )
            try:
                cam_id_front = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "camera_front_new")
                self._viewer.update_scene(self._data, camera=cam_id_front)
                img_front = self._viewer.render()
            except Exception:
                img_front = np.zeros((self._render_specs.height, self._render_specs.width, 3), dtype=np.uint8)
            try:
                cam_id_wrist = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "camera_wrist")
                self._viewer.update_scene(self._data, camera=cam_id_wrist)
                img_wrist = self._viewer.render()
            except Exception:
                img_wrist = np.zeros((self._render_specs.height, self._render_specs.width, 3), dtype=np.uint8)
            obs["pixels"] = {"front": img_front, "wrist": img_wrist}
        return obs

    def _compute_reward(self) -> float:
        success, inside = self._check_pens_in_box()
        if self.reward_type == "sparse":
            return 1.0 if success else 0.0
        return float(np.mean(inside))

    def _is_success(self) -> bool:
        success, _ = self._check_pens_in_box()
        return success

    def _get_pen_scenario(self, scenario_id: int):
        for scenario in PEN_SCENARIOS:
            if scenario["id"] == scenario_id:
                return scenario
        return None

    def _set_pens_positions(self, pens_config):
        self._data.qpos[BOX_QPOS_START:BOX_QPOS_START + 3] = BOX_POSITION
        self._data.qpos[BOX_QPOS_START + 3:BOX_QPOS_START + 7] = BOX_QUATERNION
        for pen_name, (pos, quat) in pens_config.items():
            if pen_name in PEN_QPOS_MAP:
                qpos_start = PEN_QPOS_MAP[pen_name]
                self._data.qpos[qpos_start:qpos_start + 3] = pos
                self._data.qpos[qpos_start + 3:qpos_start + 7] = quat

    def _get_body_id(self, name: str):
        body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
        return body_id if body_id >= 0 else None

    def _get_pen_geom_pos(self, pen_body_id: int):
        for geom_id in range(self._model.ngeom):
            if int(self._model.geom_bodyid[geom_id]) != pen_body_id:
                continue
            if self._model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                return self._data.geom_xpos[geom_id].copy()
        return self._data.xpos[pen_body_id].copy()

    def _in_box(self, local_pos):
        x, y, z = local_pos
        return (
            BOX_BOUNDS["x"][0] - BOX_MARGIN <= x <= BOX_BOUNDS["x"][1] + BOX_MARGIN
            and BOX_BOUNDS["y"][0] - BOX_MARGIN <= y <= BOX_BOUNDS["y"][1] + BOX_MARGIN
            and BOX_BOUNDS["z"][0] - BOX_MARGIN <= z <= BOX_BOUNDS["z"][1] + BOX_MARGIN
        )

    def _check_pens_in_box(self):
        box_id = self._get_body_id("paper_box")
        if box_id is None:
            return False, [False] * len(self._pen_names)
        box_pos = self._data.xpos[box_id].copy()
        box_mat = self._data.xmat[box_id].reshape(3, 3)
        inside = []
        for name in self._pen_names:
            pen_id = self._get_body_id(name)
            if pen_id is None:
                inside.append(False)
                continue
            pen_pos = self._get_pen_geom_pos(pen_id)
            local = box_mat.T @ (pen_pos - box_pos)
            inside.append(self._in_box(local))
        return all(inside), inside
