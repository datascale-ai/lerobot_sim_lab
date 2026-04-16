"""
SO-100 机械臂纯仿真 - JupyterLab 适配版本
可以在 Jupyter Notebook 中运行并查看渲染结果
"""

import matplotlib.pyplot as plt
import mediapy as media
import mujoco
import numpy as np
from IPython.display import Video

from lerobot_sim_lab.utils.paths import get_outputs_dir, get_so100_scene_path


class SO100Simulator:
    """SO-100 机械臂仿真器"""
    
    def __init__(self, scene="push_cube"):
        """
        初始化仿真器
        
        Args:
            scene: 场景选择
                - "basic": 基础机械臂
                - "initial": 初始位置
                - "push_cube": 推方块任务
        """
        # 选择场景文件
        xml_path = get_so100_scene_path(scene if scene in {"basic", "initial", "push_cube"} else "push_cube")
        print(f"加载场景: {xml_path}")
        
        # 加载 MuJoCo 模型
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        
        # 重置到初始关键帧
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        
        # 创建渲染器
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        
        print("✅ 仿真器初始化完成")
        print(f"   关节数: {self.model.njnt}")
        print(f"   物体数: {self.model.nbody}")
        print(f"   时间步: {self.model.opt.timestep}")
    
    def step(self, n_steps=1):
        """执行仿真步骤"""
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)
    
    def render_frame(self):
        """渲染当前帧"""
        self.renderer.update_scene(self.data)
        return self.renderer.render()
    
    def get_joint_positions(self):
        """获取关节位置"""
        return self.data.qpos[:6].copy()  # 前6个是机械臂关节
    
    def set_joint_positions(self, positions):
        """设置关节位置"""
        self.data.qpos[:6] = positions
        mujoco.mj_forward(self.model, self.data)
    
    def run_simulation(self, n_steps=200, render_every=2):
        """
        运行仿真并记录帧
        
        Args:
            n_steps: 仿真步数
            render_every: 每隔多少步渲染一次
            
        Returns:
            frames: 渲染的帧列表
            joint_positions: 关节位置历史
        """
        frames = []
        joint_positions = []
        
        print(f"运行仿真 {n_steps} 步...")
        
        for i in range(n_steps):
            # 执行仿真
            self.step()
            
            # 记录关节位置
            joint_positions.append(self.get_joint_positions())
            
            # 渲染
            if i % render_every == 0:
                frame = self.render_frame()
                frames.append(frame)
            
            if (i + 1) % 50 == 0:
                print(f"  进度: {i+1}/{n_steps}")
        
        print(f"✅ 仿真完成，渲染了 {len(frames)} 帧")
        
        return frames, np.array(joint_positions)
    
    def show_current_frame(self):
        """在 Jupyter 中显示当前帧"""
        frame = self.render_frame()
        plt.figure(figsize=(10, 6))
        plt.imshow(frame)
        plt.axis('off')
        plt.title('SO-100 Current Frame')
        plt.tight_layout()
        plt.show()
        
    def save_video(self, frames, output_path='so100_sim.mp4', fps=30):
        """保存视频"""
        media.write_video(output_path, frames, fps=fps)
        print(f"✅ 视频已保存: {output_path}")
        return output_path
    
    def plot_joint_trajectories(self, joint_positions):
        """绘制关节轨迹"""
        joint_names = [
            'Shoulder Pan', 'Shoulder Lift', 'Elbow', 
            'Wrist Pitch', 'Wrist Roll', 'Gripper'
        ]
        
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for i in range(6):
            axes[i].plot(joint_positions[:, i])
            axes[i].set_title(joint_names[i])
            axes[i].set_xlabel('Step')
            axes[i].set_ylabel('Position (rad)')
            axes[i].grid(True)
        
        plt.tight_layout()
        plt.show()


def demo_basic_simulation():
    """基础仿真演示 - 让机械臂执行推方块动作"""
    print("=" * 50)
    print("SO-100 基础仿真演示 - 推方块任务")
    print("=" * 50)
    
    # 创建仿真器
    sim = SO100Simulator(scene="push_cube")
    
    # 显示初始状态
    print("\n初始关节位置:")
    initial_pos = sim.get_joint_positions()
    print(initial_pos)
    
    # 显示当前帧
    print("\n渲染初始帧...")
    sim.show_current_frame()
    
    # 定义一个推方块的动作序列
    print("\n执行推方块动作序列...")
    
    # 动作序列：从 home 位置到推方块的几个关键姿态
    action_sequence = [
        # 阶段1: 保持初始姿态
        {"target": np.array([0, -3.14, 3.14, 0.817, 0, -0.157]), "steps": 50},
        # 阶段2: 移动到方块上方
        {"target": np.array([0.3, -2.5, 2.8, 0.5, 0, -0.157]), "steps": 100},
        # 阶段3: 下降接近方块
        {"target": np.array([0.3, -2.0, 2.5, 0.3, 0, -0.157]), "steps": 100},
        # 阶段4: 推动方块
        {"target": np.array([0.3, -2.2, 2.3, 0.2, 0, -0.157]), "steps": 150},
        # 阶段5: 抬起
        {"target": np.array([0.2, -2.8, 3.0, 0.6, 0, -0.157]), "steps": 100},
    ]
    
    frames = []
    joint_positions = []
    
    for i, action in enumerate(action_sequence):
        print(f"  阶段 {i+1}/{len(action_sequence)}: {action['steps']} 步")
        target = action["target"]
        n_steps = action["steps"]
        
        # 使用控制器设置目标位置
        sim.data.ctrl[:6] = target
        
        # 执行仿真并渲染
        for step in range(n_steps):
            sim.step()
            joint_positions.append(sim.get_joint_positions())
            
            # 每2步渲染一次
            if step % 2 == 0:
                frames.append(sim.render_frame())
    
    joint_positions = np.array(joint_positions)
    print(f"\n✅ 仿真完成，渲染了 {len(frames)} 帧")
    
    # 绘制关节轨迹
    print("\n关节轨迹:")
    sim.plot_joint_trajectories(joint_positions)
    
    # 保存视频
    video_path = sim.save_video(frames, output_path=str(get_outputs_dir() / 'so100_simulation.mp4'), fps=30)
    
    # 在 Jupyter 中显示视频
    print("\n视频预览:")
    return Video(video_path, embed=True, width=640)


def demo_interactive_control():
    """交互式控制演示 - 展示平滑运动"""
    print("=" * 50)
    print("SO-100 交互式控制演示")
    print("=" * 50)
    
    sim = SO100Simulator(scene="push_cube")
    
    # 定义一些目标姿态（使用控制器）
    poses = {
        "home": np.array([0, -3.14, 3.14, 0.817, 0, -0.157]),
        "reach_right": np.array([0.8, -2.5, 2.8, 0.6, 0, -0.157]),
        "reach_left": np.array([-0.8, -2.5, 2.8, 0.6, 0, -0.157]),
        "wave": np.array([0, -2.0, 2.5, 0.3, 1.5, -0.157]),
        "close_gripper": np.array([0, -3.14, 3.14, 0.817, 0, 0.8]),
    }
    
    all_frames = []
    
    for pose_name, target_pos in poses.items():
        print(f"\n移动到 {pose_name} 姿态...")
        print(f"目标位置: {target_pos}")
        
        # 使用控制器控制（更平滑）
        sim.data.ctrl[:6] = target_pos
        
        # 运行仿真让机械臂移动到目标位置
        for step in range(80):
            sim.step()
            if step % 2 == 0:
                all_frames.append(sim.render_frame())
    
    print(f"\n✅ 生成了 {len(all_frames)} 帧")
    
    # 保存整个序列
    video_path = sim.save_video(all_frames, output_path=str(get_outputs_dir() / 'so100_control.mp4'), fps=30)
    
    print("\n控制序列视频:")
    return Video(video_path, embed=True, width=640)


if __name__ == "__main__":
    # 如果直接运行脚本
    print("请在 Jupyter Notebook 中导入此模块并调用:")
    print("  from lerobot_sim_lab.sim.simulator import demo_basic_simulation, demo_interactive_control")
    print("  demo_basic_simulation()")
    print("  demo_interactive_control()")
