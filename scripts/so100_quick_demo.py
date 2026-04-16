"""
SO-100 快速演示脚本 - 可以直接在 JupyterLab 中运行
"""

import numpy as np
from IPython.display import Video
import matplotlib.pyplot as plt

from lerobot_sim_lab.sim.simulator import SO100Simulator, demo_basic_simulation


def demo_free_fall():
    """演示1：自由下落（无控制）"""
    print("=" * 60)
    print("演示1：自由下落（关闭控制器）")
    print("=" * 60)
    
    # 创建基础场景
    sim = SO100Simulator(scene="basic")
    
    # 设置初始位置（抬起状态）
    initial_pos = np.array([0.5, -2.0, 2.5, 1.0, 0.5, 0])
    sim.set_joint_positions(initial_pos)
    
    # 关闭控制器（让机械臂自由下落）
    sim.model.actuator_gainprm[:, 0] = 0  # 关闭 kp
    
    print("初始位置:", initial_pos)
    print("运行自由下落仿真...")
    
    # 运行仿真
    frames, joints = sim.run_simulation(n_steps=300, render_every=2)
    
    # 绘制轨迹
    sim.plot_joint_trajectories(joints)
    
    # 保存视频
    video_path = sim.save_video(frames, output_path='outputs/free_fall.mp4', fps=30)
    print(f"✅ 视频保存: {video_path}")
    
    return Video(video_path, embed=True, width=640)


def demo_controlled_motion():
    """演示2：受控运动（推方块）"""
    print("\n" + "=" * 60)
    print("演示2：受控运动（推方块任务）")
    print("=" * 60)
    
    return demo_basic_simulation()


def demo_smooth_trajectory():
    """演示3：平滑轨迹（正弦波）"""
    print("\n" + "=" * 60)
    print("演示3：平滑正弦波轨迹")
    print("=" * 60)
    
    sim = SO100Simulator(scene="push_cube")
    
    frames = []
    joint_history = []
    
    print("生成正弦波轨迹...")
    
    # 让肩部关节做正弦摆动
    for i in range(300):
        t = i / 50.0
        
        # 基础姿态 + 正弦波
        target = np.array([
            0.8 * np.sin(t * 2 * np.pi),          # 肩部旋转摆动
            -3.0 + 0.5 * np.sin(t * 2 * np.pi),   # 肩部俯仰
            3.0 + 0.3 * np.cos(t * 2 * np.pi),    # 肘部
            0.6,                                   # 腕部俯仰
            0.5 * np.sin(t * 4 * np.pi),          # 腕部快速旋转
            -0.157                                 # 夹爪
        ])
        
        # 设置控制器目标
        sim.data.ctrl[:6] = target
        
        # 执行仿真
        sim.step(3)
        
        # 记录
        joint_history.append(sim.get_joint_positions())
        if i % 2 == 0:
            frames.append(sim.render_frame())
    
    joint_history = np.array(joint_history)
    print(f"生成了 {len(frames)} 帧")
    
    # 绘制前3个关节的轨迹
    plt.figure(figsize=(14, 4))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.plot(joint_history[:, i])
        plt.title(['Shoulder Pan', 'Shoulder Lift', 'Elbow'][i])
        plt.xlabel('Step')
        plt.ylabel('Position (rad)')
        plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # 保存视频
    video_path = sim.save_video(frames, output_path='outputs/sine_wave.mp4', fps=25)
    print(f"✅ 视频保存: {video_path}")
    
    return Video(video_path, embed=True, width=640)


def demo_all():
    """运行所有演示"""
    print("🚀 运行所有演示\n")
    
    print("=" * 70)
    v1 = demo_free_fall()
    
    print("\n" + "=" * 70)
    v2 = demo_controlled_motion()
    
    print("\n" + "=" * 70)
    v3 = demo_smooth_trajectory()
    
    print("\n" + "=" * 70)
    print("✅ 所有演示完成！")
    print("视频保存在: outputs/")
    
    return v1, v2, v3


if __name__ == "__main__":
    print("请在 Jupyter Notebook 中运行:")
    print()
    print("from scripts.so100_quick_demo import demo_free_fall, demo_controlled_motion, demo_smooth_trajectory, demo_all")
    print()
    print("# 运行单个演示:")
    print("video1 = demo_free_fall()")
    print("video1  # 显示视频")
    print()
    print("# 或运行全部:")
    print("demo_all()")
