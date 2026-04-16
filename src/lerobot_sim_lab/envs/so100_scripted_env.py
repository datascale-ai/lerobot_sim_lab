"""Scripted SO-100 pick-and-place environment for headless data collection."""

from typing import Any

import mujoco
import numpy as np

from lerobot_sim_lab.config.scenarios.pick_place import get_action_sequence, get_scenario
from lerobot_sim_lab.envs.so100_gym_env import SO100PickCubeGymEnv


class SO100PickCubeScriptedEnv(SO100PickCubeGymEnv):
    """
    SO-100 机械臂抓取任务环境 - 自动执行版本
    
    特点：
    - ✅ 使用预定义的动作序列（来自 demo_pick_and_place.py）
    - ✅ 无需键盘/手柄交互
    - ✅ 适用于 Docker 无窗口环境
    - ✅ 可直接录制成 LeRobot 格式
    
    工作流程：
    1. 环境自动执行预定义的动作序列
    2. 每次调用 step() 都会执行序列中的下一个动作
    3. 当序列执行完毕时，episode 结束
    """

    def __init__(
        self,
        render_mode: str | None = None,
        observation_width: int = 640,
        observation_height: int = 480,
        image_obs: bool = True,  # 默认开启图像观测
        reward_type: str = "sparse",
        success_threshold: float = 0.05,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        use_gripper: bool = True,  # gym_manipulator 传递的参数
        gripper_penalty: float = 0.0,  # gym_manipulator 传递的参数
        **kwargs,  # 接受其他可能的参数
    ):
        """
        初始化自动执行版本的SO-100抓取环境
        
        Args:
            use_gripper: 是否使用夹爪（兼容参数，暂不使用）
            gripper_penalty: 夹爪惩罚（兼容参数，暂不使用）
        """
        super().__init__(
            render_mode=render_mode,
            observation_width=observation_width,
            observation_height=observation_height,
            image_obs=image_obs,
            reward_type=reward_type,
            success_threshold=success_threshold,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
        )
        
        # 场景配置（从场景库动态加载）
        self._current_scenario_id = 1  # 默认场景1
        self._action_sequence = get_action_sequence(self._current_scenario_id)
        
        # 当前执行状态
        self._current_action_idx = 0  # 当前执行到哪个动作
        self._current_step_in_action = 0  # 当前动作内的步数
        self._total_steps = sum(steps for _, steps, _ in self._action_sequence)
        self._episode_step = 0
        
        # 轨迹插值：记录起始和目标姿态
        self._start_qpos = None  # 当前阶段的起始姿态
        self._target_qpos = None  # 当前阶段的目标姿态
        
        # 计算子步数 (每个控制步包含多少个物理步)
        self._n_substeps = int(control_dt / physics_dt)

    def reset(self, seed=None, options=None, **kwargs) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """
        重置环境并加载指定场景
        
        Args:
            options: 可选参数，包含:
                - scenario_idx: 场景ID (1, 2, ..., 默认1)
        """
        # 解析场景ID
        if options is not None and 'scenario_idx' in options:
            scenario_idx = options['scenario_idx']
        else:
            scenario_idx = 1  # 默认场景1
        
        # 加载场景配置
        try:
            scenario = get_scenario(scenario_idx)
            self._current_scenario_id = scenario_idx
            self._action_sequence = get_action_sequence(scenario_idx)
            self._total_steps = sum(steps for _, steps, _ in self._action_sequence)
            
            # 设置cube初始位置（在父类reset之前）
            cube_pos = scenario['cube_pos']
        except Exception as e:
            print(f"⚠️  加载场景 {scenario_idx} 失败: {e}")
            print("    使用默认场景1")
            scenario = get_scenario(1)
            self._current_scenario_id = 1
            self._action_sequence = get_action_sequence(1)
            self._total_steps = sum(steps for _, steps, _ in self._action_sequence)
            cube_pos = scenario['cube_pos']
        
        # 调用父类reset
        obs, info = super().reset(seed=seed, **kwargs)
        
        # 设置cube位置（父类reset后会初始化物理状态）
        self._data.qpos[6:9] = cube_pos
        self._data.qpos[9:13] = [1, 0, 0, 0]  # 四元数姿态
        mujoco.mj_forward(self._model, self._data)
        
        # 重置动作序列状态
        self._current_action_idx = 0
        self._current_step_in_action = 0
        self._episode_step = 0
        
        # 重置插值状态
        self._start_qpos = None
        self._target_qpos = None
        
        # 更新info
        info["scenario_id"] = self._current_scenario_id
        info["scenario_name"] = scenario['name']
        info["cube_position"] = cube_pos.tolist()
        info["action_sequence_progress"] = f"0/{len(self._action_sequence)}"
        info["episode_progress"] = f"0/{self._total_steps}"
        
        return obs, info
    
    def step(self, action: np.ndarray | None = None) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """
        执行一步动作
        
        注意：传入的 action 参数会被忽略，环境使用预定义的动作序列
        """
        # 检查是否已执行完所有动作
        if self._current_action_idx >= len(self._action_sequence):
            # Episode 结束
            obs = self._compute_observation()
            reward = self._compute_reward()
            success = self._is_success()
            
            if self.reward_type == "sparse":
                reward = 1.0 if success else 0.0
            
            info = {
                "succeed": success,
                "action_sequence_complete": True,
                "action_sequence_progress": f"{len(self._action_sequence)}/{len(self._action_sequence)}",
                "episode_progress": f"{self._episode_step}/{self._total_steps}",
            }
            
            return obs, reward, True, False, info
        
        # 获取当前应执行的动作
        target_qpos, steps_in_action, description = self._action_sequence[self._current_action_idx]
        
        # === 轨迹插值逻辑 ===
        # 当进入新阶段时，记录起始和目标姿态
        if self._current_step_in_action == 0:
            # 起始姿态：使用当前实际关节位置
            self._start_qpos = self._data.qpos[:6].copy()
            self._target_qpos = target_qpos.copy()
        
        # 计算插值比例 (0.0 -> 1.0)
        alpha = (self._current_step_in_action + 1) / steps_in_action
        
        # 线性插值：从起始姿态平滑过渡到目标姿态
        interpolated_qpos = (1.0 - alpha) * self._start_qpos + alpha * self._target_qpos
        
        # 设置插值后的控制目标（平滑的轨迹）
        self._data.ctrl[:6] = interpolated_qpos
        
        # 执行 MuJoCo 仿真步 (模拟 control_dt 时间)
        for _ in range(self._n_substeps):
            mujoco.mj_step(self._model, self._data)
        
        # 更新步数
        self._current_step_in_action += 1
        self._episode_step += 1
        
        # 检查当前动作是否执行完毕
        if self._current_step_in_action >= steps_in_action:
            self._current_action_idx += 1
            self._current_step_in_action = 0
        
        # 计算观测、奖励和终止条件
        obs = self._compute_observation()
        reward = self._compute_reward()
        success = self._is_success()
        
        if self.reward_type == "sparse":
            reward = 1.0 if success else 0.0
        
        # 检查cube是否超出边界或掉落（仅记录，不提前终止）
        cube_pos = self._get_cube_position()
        out_of_bounds = np.any(cube_pos[:2] < -0.15) or np.any(cube_pos[:2] > 0.15)
        cube_dropped = cube_pos[2] < 0.01
        
        # 脚本化环境不提前终止，让整个序列执行完
        # 这样可以保证数据长度一致性
        info = {
            "succeed": success,
            "out_of_bounds": out_of_bounds,
            "cube_dropped": cube_dropped,
            "current_action": description,
            "action_sequence_progress": f"{self._current_action_idx}/{len(self._action_sequence)}",
            "episode_progress": f"{self._episode_step}/{self._total_steps}",
            "action_sequence_complete": self._current_action_idx >= len(self._action_sequence),
        }
        
        # 不提前终止，让序列完整执行
        terminated = False
        
        return obs, reward, terminated, False, info

    def get_scripted_action(self) -> np.ndarray:
        """
        获取当前应执行的脚本化动作（供外部调用）
        
        Returns:
            6D joint position target (插值后的平滑轨迹点)
        """
        if self._current_action_idx >= len(self._action_sequence):
            # 返回最后一个动作
            return self._action_sequence[-1][0].copy()
        
        # 返回当前的插值姿态（如果已初始化）
        if self._start_qpos is not None and self._target_qpos is not None:
            target_qpos, steps_in_action, _ = self._action_sequence[self._current_action_idx]
            alpha = (self._current_step_in_action + 1) / steps_in_action
            interpolated_qpos = (1.0 - alpha) * self._start_qpos + alpha * self._target_qpos
            return interpolated_qpos.copy()
        else:
            # 第一步，还未初始化插值状态
            target_qpos, _, _ = self._action_sequence[self._current_action_idx]
            return target_qpos.copy()
