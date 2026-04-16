"""
抓取并放置演示 - 机械臂抓取cube从黄色区域放到紫色区域
"""
import sys

import numpy as np
import mujoco
import mediapy as media
from IPython.display import Video, display
import matplotlib.pyplot as plt

from lerobot_sim_lab.utils.paths import get_so100_scene_path, resolve_output_path

# 使用默认英文字体
plt.rcParams['font.family'] = 'DejaVu Sans'
def analyze_pick_place_task():
    """分析抓取放置任务的可行性"""
    print("=" * 80)
    print("抓取放置任务分析")
    print("=" * 80)
    
    model = mujoco.MjModel.from_xml_path(str(get_so100_scene_path("push_cube")))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    # 位置信息
    cube_pos = data.qpos[6:9]
    cube_size = 0.015  # cube半径
    
    print(f"\n任务目标:")
    print(f"  起始位置 (黄色区域): X = 0.06 m")
    print(f"  目标位置 (紫色区域): X = -0.06 m")
    print(f"  Cube 当前位置: ({cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f})")
    print(f"  Cube 尺寸: {cube_size*2*1000:.1f} mm × {cube_size*2*1000:.1f} mm × {cube_size*2*1000:.1f} mm")
    print(f"  需要移动距离: {0.12:.3f} m = {120:.0f} mm")
    
    # 夹爪信息
    jaw_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "Jaw")
    jaw_range = model.jnt_range[jaw_joint_id]
    
    print(f"\n夹爪信息:")
    print(f"  Jaw 关节范围: [{jaw_range[0]:.3f}, {jaw_range[1]:.3f}] rad")
    print(f"                [{np.rad2deg(jaw_range[0]):.1f}°, {np.rad2deg(jaw_range[1]):.1f}°]")
    print(f"  闭合 (抓取): {jaw_range[0]:.3f} rad")
    print(f"  张开 (释放): {jaw_range[1]:.3f} rad")
    
    print(f"\n任务步骤:")
    steps = [
        "1. 移动到 cube 上方",
        "2. 下降到 cube 高度",
        "3. 张开夹爪",
        "4. 前进包围 cube",
        "5. 闭合夹爪 (抓取)",
        "6. 抬起 cube",
        "7. 移动到目标位置上方",
        "8. 下降到目标高度",
        "9. 张开夹爪 (释放)",
        "10. 抬起并返回",
    ]
    
    for step in steps:
        print(f"  {step}")
    
    print("=" * 80)


def demo_pick_and_place():
    """完整的抓取并放置演示"""
    # 确保输出目录存在
    import os
    os.makedirs('outputs', exist_ok=True)
    
    print("\n" + "=" * 80)
    print("抓取并放置演示")
    print("=" * 80)
    
    model = mujoco.MjModel.from_xml_path(str(get_so100_scene_path("push_cube")))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    renderer = mujoco.Renderer(model, height=480, width=640)
    
    gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Fixed_Jaw")
    
    frames_side = []
    frames_front = []
    cube_positions = []
    gripper_positions = []
    jaw_angles = []
    
    # ⭐ 使用 joint_range_explorer 找到的最佳配置
    # 最佳配置: Rotation=-0.25, Pitch=-0.6789, Elbow=1.4874, Wrist_Pitch=0.5333
    # 末端位置: (0.0610, 0.1339, 0.0154), 距离cube仅2.16mm！
    
    best_pick_config = np.array([-0.2500, -0.6789, 1.4874, 0.5333, 0.0, -0.157])
    
    # 精心设计的抓取-放置动作序列
    # 格式: (6个关节角度, 步数, 描述)
    actions = [
        # === 阶段1: 准备 ===
        (np.array([0, -3.14, 3.14, 0.817, 0, -0.157]), 30, "初始姿态"),
        
        # === 阶段2: 移动到cube上方 ===
        (np.array([-0.25, -2.5, 2.5, 0.6, 0, -0.157]), 60, "转向黄色区域"),
        
        # === 阶段3: 逐步下降 ===
        (np.array([-0.25, -1.6, 2.3, 0.5, 0, 0.5]), 50, "下降中..."),
        (np.array([-0.25, -1.45, 2.2, 0.5, 0, 0.6]), 50, "继续下降..."),
        
        # === 阶段4: 张开夹爪准备抓取 ===
        (np.array([-0.25, -1.35, 2.2, 0.5, 0, 0.65]), 40, "张开夹爪"),
        
        # === 阶段6: 闭合夹爪 - 抓取！===
        (np.array([-0.25, -1.25, 2.2, 0.5, 0, 0.1]), 60, "闭合夹爪 - 抓取！"),
        
        # === 阶段7: 抬起cube ===
        (np.array([-0.25, -1.75, 2.1, 0.5, 0, 0.1]), 60, "抬起cube"),
        (np.array([-0.25, -2.0, 2.1, 0.5, 0, 0.1]), 60, "继续抬高"),
        
        # === 阶段8: 移动到目标位置上方 (镜像到紫色区域) ===
        (np.array([0.0, -2.0, 2.2, 0.5, 0, 0.2]), 60, "移动到中央"),
        (np.array([0.25, -1.8, 2.3, 0.5, 0, 0.2]), 60, "移动到紫色区域上方"),
        
        # === 阶段10: 张开夹爪 - 释放！===
        (np.array([0.25, -1.8, 2.35, 0.5, 0, 0.6]), 60, "张开夹爪 - 释放！"),
        
        # === 阶段11: 后退并抬起 ===
        (np.array([0.25, -1.5, 2.0, 0.6, 0, 1.5]), 50, "后退"),
        (np.array([0.25, -2.5, 2.5, 0.7, 0, -0.157]), 50, "抬起"),
        
        # === 阶段12: 返回初始姿态 ===
        (np.array([0, -2.8, 2.8, 0.7, 0, -0.157]), 50, "移动到中央"),
        (np.array([0, -3.14, 3.14, 0.817, 0, -0.157]), 60, "回到初始姿态"),
    ]
    
    print("\n执行动作序列:")
    print(f"  总共 {len(actions)} 个阶段\n")
    
    for i, (target, steps, desc) in enumerate(actions):
        print(f"  阶段 {i+1:2d}/{len(actions)}: {desc:30s} ({steps:3d}步)")
        print(f"    Action Target: {np.round(target, 4)}")
        data.ctrl[:6] = target

        for step in range(steps):
            mujoco.mj_step(model, data)

            # 记录数据
            cube_pos = data.qpos[6:9].copy()
            gripper_pos = data.xpos[gripper_id].copy()
            jaw_angle = data.qpos[5]  # 第6个关节

            cube_positions.append(cube_pos)
            gripper_positions.append(gripper_pos)
            jaw_angles.append(jaw_angle)

            # 渲染 (双视角：侧面视图+正面视图)
            if step % 2 == 0:
                # 侧面视图
                cam_side = mujoco.MjvCamera()
                cam_side.lookat = np.array([0, 0.08, 0.15])
                cam_side.distance = 0.6
                cam_side.azimuth = 0    # 侧面
                cam_side.elevation = -10
                renderer.update_scene(data, camera=cam_side)
                frame_side = renderer.render()

                # 正面视图
                cam_front = mujoco.MjvCamera()
                cam_front.lookat = np.array([0, 0.08, 0.15])
                cam_front.distance = 0.6
                cam_front.azimuth = 270     # 正面
                cam_front.elevation = -10
                renderer.update_scene(data, camera=cam_front)
                frame_front = renderer.render()

                frames_side.append(frame_side)
                frames_front.append(frame_front)

    
    cube_positions = np.array(cube_positions)
    gripper_positions = np.array(gripper_positions)
    jaw_angles = np.array(jaw_angles)
    
    # 计算结果
    start_pos = cube_positions[0]
    end_pos = cube_positions[-1]
    total_movement = np.linalg.norm(end_pos[:2] - start_pos[:2])
    target_x = -0.06
    final_error = abs(end_pos[0] - target_x)
    
    print(f"\n{'='*80}")
    print("任务结果:")
    print(f"{'='*80}")
    print(f"  Cube 起始位置: ({start_pos[0]:.4f}, {start_pos[1]:.4f}, {start_pos[2]:.4f})")
    print(f"  Cube 结束位置: ({end_pos[0]:.4f}, {end_pos[1]:.4f}, {end_pos[2]:.4f})")
    print(f"  移动距离: {total_movement*1000:.1f} mm")
    print(f"  目标位置: X = {target_x:.4f}")
    print(f"  位置误差: {final_error*1000:.1f} mm")
    
    success = total_movement > 0.05 and final_error < 0.03
    print(f"\n  任务状态: {'✅ 成功！' if success else '⚠️  部分成功/需要调整'}")
    
    if not success:
        if total_movement < 0.05:
            print(f"    原因: Cube 移动距离不足 ({total_movement*1000:.1f} mm < 50 mm)")
            print(f"    建议: 检查夹爪是否真正抓住了 cube")
        elif final_error > 0.03:
            print(f"    原因: Cube 未到达目标位置 (误差 {final_error*1000:.1f} mm)")
            print(f"    建议: 调整移动阶段的关节角度")
    
    print(f"{'='*80}")
    
    # 绘制分析图
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Cube X trajectory
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(cube_positions[:, 0], linewidth=2, color='blue')
    ax1.axhline(y=0.06, color='orange', linestyle='--', linewidth=2, label='Start (yellow zone)')
    ax1.axhline(y=-0.06, color='purple', linestyle='--', linewidth=2, label='Target (purple zone)')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Cube X Position (m)')
    ax1.set_title('Cube X Movement Trajectory')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Cube height
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(cube_positions[:, 2] * 1000, linewidth=2, color='green')
    ax2.axhline(y=17, color='red', linestyle='--', alpha=0.5, label='Initial height')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Cube Height (mm)')
    ax2.set_title('Cube Height Change (Lift Detection)')
    ax2.legend()
    ax2.grid(True)
    
    # 3. Gripper angle
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(np.rad2deg(jaw_angles), linewidth=2, color='red')
    ax3.axhline(y=np.rad2deg(-0.2), color='blue', linestyle='--', label='Closed (grasp)')
    ax3.axhline(y=np.rad2deg(1.5), color='orange', linestyle='--', label='Open (release)')
    ax3.set_xlabel('Steps')
    ax3.set_ylabel('Gripper Angle (deg)')
    ax3.set_title('Gripper Open/Close State')
    ax3.legend()
    ax3.grid(True)
    
    # 4. Distance to target
    ax4 = plt.subplot(3, 2, 4)
    distances = np.abs(cube_positions[:, 0] - target_x) * 1000
    ax4.plot(distances, linewidth=2, color='purple')
    ax4.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Success threshold (30mm)')
    ax4.set_xlabel('Steps')
    ax4.set_ylabel('Distance to Target (mm)')
    ax4.set_title('Distance to Purple Zone')
    ax4.legend()
    ax4.grid(True)
    
    # 5. Top-down view trajectory
    ax5 = plt.subplot(3, 2, 5)
    
    # Cube trajectory
    scatter = ax5.scatter(cube_positions[:, 0], cube_positions[:, 1], 
                         c=range(len(cube_positions)), cmap='viridis', 
                         s=20, alpha=0.6)
    plt.colorbar(scatter, ax=ax5, label='Time step')
    
    # Start and end positions
    ax5.plot(start_pos[0], start_pos[1], 'go', markersize=20, 
            markeredgecolor='black', linewidth=3, label='Start', zorder=10)
    ax5.plot(end_pos[0], end_pos[1], 'ro', markersize=20, 
            markeredgecolor='black', linewidth=3, label='End', zorder=10)
    
    # Target zones
    from matplotlib.patches import Rectangle
    goal1 = Rectangle((0.025, 0.09), 0.07, 0.09, 
                      facecolor='yellow', alpha=0.3, edgecolor='orange', linewidth=2)
    goal2 = Rectangle((-0.095, 0.09), 0.07, 0.09, 
                      facecolor='magenta', alpha=0.3, edgecolor='purple', linewidth=2)
    ax5.add_patch(goal1)
    ax5.add_patch(goal2)
    
    # Fence
    ax5.plot([-0.125, -0.125], [0.09, 0.18], 'k-', linewidth=3)
    ax5.plot([0.125, 0.125], [0.09, 0.18], 'k-', linewidth=3)
    ax5.plot([-0.125, 0.125], [0.09, 0.09], 'k-', linewidth=3)
    ax5.plot([-0.125, 0.125], [0.18, 0.18], 'k-', linewidth=3)
    
    ax5.set_xlabel('X Position (m)')
    ax5.set_ylabel('Y Position (m)')
    ax5.set_title('Cube Movement Trajectory (Top View)')
    ax5.legend()
    ax5.axis('equal')
    ax5.grid(True, alpha=0.3)
    
    # 6. Side view (Y-Z)
    ax6 = plt.subplot(3, 2, 6)
    ax6.plot(cube_positions[:, 1], cube_positions[:, 2], 
            linewidth=2, color='blue', label='Cube trajectory')
    ax6.plot(start_pos[1], start_pos[2], 'go', markersize=15, 
            markeredgecolor='black', linewidth=2, label='Start')
    ax6.plot(end_pos[1], end_pos[2], 'ro', markersize=15, 
            markeredgecolor='black', linewidth=2, label='End')
    ax6.axhline(y=0, color='brown', linestyle='--', linewidth=2, label='Ground')
    ax6.set_xlabel('Y Position (m)')
    ax6.set_ylabel('Z Height (m)')
    ax6.set_title('Side View: Cube Height Trajectory')
    ax6.legend()
    ax6.grid(True)
    ax6.set_aspect('equal')
    
    plt.tight_layout()
    analysis_path = resolve_output_path("pick_and_place_analysis.png")
    plt.savefig(analysis_path, dpi=150, bbox_inches='tight')
    print(f"\n分析图保存: {analysis_path}")
    # plt.show() 在非交互环境下不需要调用
    plt.close()
    
    # 保存视频
    video_path_side = str(resolve_output_path("pick_and_place_demo_side_1.mp4"))
    video_path_front = str(resolve_output_path("pick_and_place_demo_front_1.mp4"))
    media.write_video(video_path_side, frames_side, fps=30)
    media.write_video(video_path_front, frames_front, fps=30)
    print(f"视频保存: {resolve_output_path('')}")
    
    # 在 Jupyter 中显示视频（可选）
    try:
        from IPython.display import display
        print("--- 侧面视角结果 (Side View) ---")
        side_video = Video(video_path_side, embed=True, width=640)
        display(side_video)

        print("\n--- 正面视角结果 (Front View) ---")
        front_video = Video(video_path_front, embed=True, width=640)
        display(front_video)
    except:
        # 非 Jupyter 环境，跳过 display
        pass
    
    # 返回视频路径
    return video_path_front, video_path_side

if __name__ == "__main__":
    import sys
    
    # 支持命令行参数选择运行哪个演示
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = "pick"  # 默认运行抓取演示
    
    print("=" * 80)
    print("SO-100 Pick and Place Demo")
    print("=" * 80)
    print()
    
    if mode == "analyze":
        print("运行任务分析...")
        analyze_pick_place_task()
    else:
        print("运行抓取并放置演示...")
        video_front, video_side = demo_pick_and_place()
    print()
    print("✅ 演示完成！视频已保存：")
    print(f"   - 正面视角: {video_front}")
    print(f"   - 侧面视角: {video_side}")
    print()
    print("提示: 可以使用以下命令运行其他演示:")
    print("  python3 demo_pick_and_place.py analyze  # 任务分析")
