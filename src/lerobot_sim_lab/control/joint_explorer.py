"""
关节范围探索器 - 查看和测试每个关节的活动范围
"""

import matplotlib.pyplot as plt
import mediapy as media
import mujoco
import numpy as np
from IPython.display import Video

from lerobot_sim_lab.utils.paths import get_outputs_dir, get_so100_models_dir


def show_joint_ranges():
    """显示所有关节的活动范围"""
    print("=" * 80)
    print("SO-100 机械臂关节活动范围")
    print("=" * 80)
    
    model = mujoco.MjModel.from_xml_path((get_so100_models_dir() / 'push_cube_loop.xml').as_posix())
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    # 关节名称和描述
    joint_info = {
        0: ("Rotation", "肩部旋转", "左右转动"),
        1: ("Pitch", "肩部俯仰", "上下摆动"),
        2: ("Elbow", "肘部", "弯曲伸直"),
        3: ("Wrist_Pitch", "腕部俯仰", "手腕上下"),
        4: ("Wrist_Roll", "腕部旋转", "手腕转动"),
        5: ("Jaw", "夹爪", "开合"),
    }
    
    print("\n关节详细信息:\n")
    print(f"{'序号':<4} {'名称':<15} {'中文':<12} {'范围 (弧度)':<25} {'范围 (度)':<25} {'描述':<10}")
    print("-" * 100)
    
    for i in range(6):
        name, cn_name, desc = joint_info[i]
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        
        # 获取关节范围
        jnt_range = model.jnt_range[jnt_id]
        min_rad, max_rad = jnt_range[0], jnt_range[1]
        min_deg, max_deg = np.rad2deg(min_rad), np.rad2deg(max_rad)
        
        print(f"{i:<4} {name:<15} {cn_name:<12} [{min_rad:6.3f}, {max_rad:6.3f}]    "
              f"[{min_deg:7.1f}°, {max_deg:7.1f}°]    {desc:<10}")
    
    print("\n" + "=" * 80)
    print("初始姿态 (home):")
    home_pose = np.array([0, -3.14, 3.14, 0.817, 0, -0.157])
    for i in range(6):
        name, cn_name, _ = joint_info[i]
        print(f"  {i}. {name:15} ({cn_name:12}): {home_pose[i]:7.3f} rad = {np.rad2deg(home_pose[i]):7.1f}°")
    
    print("=" * 80)
    
    return model, data, joint_info


def test_single_joint(joint_idx, model=None, data=None):
    """测试单个关节在整个范围内的运动"""
    if model is None:
        model = mujoco.MjModel.from_xml_path((get_so100_models_dir() / 'push_cube_loop.xml').as_posix())
        data = mujoco.MjData(model)
    
    joint_names = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
    joint_name = joint_names[joint_idx]
    
    print(f"\n测试关节 {joint_idx}: {joint_name}")
    print("-" * 60)
    
    # 获取关节范围
    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    jnt_range = model.jnt_range[jnt_id]
    
    # 获取末端执行器和cube
    gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Fixed_Jaw")
    cube_pos = data.qpos[6:9]
    
    # 在关节范围内采样
    n_samples = 20
    joint_values = np.linspace(jnt_range[0], jnt_range[1], n_samples)
    
    gripper_positions = []
    distances_to_cube = []
    heights = []
    
    # 基准姿态（其他关节保持不变）
    base_pose = np.array([0, -3.14, 3.14, 0.817, 0, -0.157])
    
    for val in joint_values:
        test_pose = base_pose.copy()
        test_pose[joint_idx] = val
        
        data.qpos[:6] = test_pose
        mujoco.mj_forward(model, data)
        
        gripper_pos = data.xpos[gripper_id].copy()
        gripper_positions.append(gripper_pos)
        
        distance = np.linalg.norm(gripper_pos - cube_pos)
        distances_to_cube.append(distance)
        heights.append(gripper_pos[2])
    
    gripper_positions = np.array(gripper_positions)
    distances_to_cube = np.array(distances_to_cube)
    heights = np.array(heights)
    
    # 找到最接近cube的位置
    min_dist_idx = np.argmin(distances_to_cube)
    min_distance = distances_to_cube[min_dist_idx]
    best_value = joint_values[min_dist_idx]
    
    print(f"  范围: [{jnt_range[0]:.3f}, {jnt_range[1]:.3f}] rad")
    print(f"  范围: [{np.rad2deg(jnt_range[0]):.1f}°, {np.rad2deg(jnt_range[1]):.1f}°]")
    print(f"  最接近cube的值: {best_value:.3f} rad ({np.rad2deg(best_value):.1f}°)")
    print(f"  最小距离: {min_distance*1000:.1f} mm")
    print(f"  末端高度范围: [{np.min(heights)*1000:.1f}, {np.max(heights)*1000:.1f}] mm")
    print(f"  Cube 高度: {cube_pos[2]*1000:.1f} mm")
    
    return joint_values, gripper_positions, distances_to_cube, heights


def find_best_configuration():
    """搜索能够触及cube的最佳关节配置"""
    print("\n" + "=" * 80)
    print("搜索最佳关节配置")
    print("=" * 80)
    
    model = mujoco.MjModel.from_xml_path((get_so100_models_dir() / 'push_cube_loop.xml').as_posix())
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Fixed_Jaw")
    cube_pos = data.qpos[6:9]
    
    print(f"\n目标: Cube @ ({cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f})")
    print(f"      高度 = {cube_pos[2]*1000:.1f} mm\n")
    
    # 获取关节范围
    joint_ranges = []
    joint_names = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
    
    for name in joint_names:
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        joint_ranges.append(model.jnt_range[jnt_id].copy())
    
    # 网格搜索策略
    print("搜索策略: 针对关键关节进行密集搜索\n")
    
    # 关键关节：Pitch (1) 和 Elbow (2) 控制高度
    # Rotation (0) 控制左右
    # Wrist_Pitch (3) 微调角度
    
    best_config = None
    best_distance = float('inf')
    
    # 搜索范围
    rotation_samples = np.linspace(-0.5, 0.5, 5)  # 小范围左右
    pitch_samples = np.linspace(-3.14, 0.2, 20)   # 全范围上下
    elbow_samples = np.linspace(0, 3.14, 20)      # 全范围弯曲
    wrist_samples = np.linspace(-2.0, 1.8, 10)    # 较大范围
    
    print(f"搜索空间: {len(rotation_samples)} × {len(pitch_samples)} × {len(elbow_samples)} × {len(wrist_samples)}")
    print(f"           = {len(rotation_samples) * len(pitch_samples) * len(elbow_samples) * len(wrist_samples)} 个配置")
    print("\n开始搜索...")
    
    tested = 0
    total = len(rotation_samples) * len(pitch_samples) * len(elbow_samples) * len(wrist_samples)
    
    for rot in rotation_samples:
        for pitch in pitch_samples:
            for elbow in elbow_samples:
                for wrist in wrist_samples:
                    tested += 1
                    
                    # 测试配置
                    test_pose = np.array([rot, pitch, elbow, wrist, 0, -0.157])
                    data.qpos[:6] = test_pose
                    mujoco.mj_forward(model, data)
                    
                    gripper_pos = data.xpos[gripper_id]
                    distance = np.linalg.norm(gripper_pos - cube_pos)
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_config = test_pose.copy()
                        best_gripper_pos = gripper_pos.copy()
                    
                    # 进度
                    if tested % 2000 == 0:
                        print(f"  进度: {tested}/{total} ({tested*100/total:.1f}%) - 当前最佳: {best_distance*1000:.1f} mm")
    
    print("\n✅ 搜索完成！")
    print(f"\n{'='*80}")
    print("最佳配置:")
    print(f"{'='*80}")
    
    joint_info = [
        ("Rotation", "肩部旋转"),
        ("Pitch", "肩部俯仰"),  
        ("Elbow", "肘部"),
        ("Wrist_Pitch", "腕部俯仰"),
        ("Wrist_Roll", "腕部旋转"),
        ("Jaw", "夹爪"),
    ]
    
    for i, (name, cn_name) in enumerate(joint_info):
        print(f"  {i}. {name:15} ({cn_name:12}): {best_config[i]:7.4f} rad = {np.rad2deg(best_config[i]):8.2f}°")
    
    print("\n结果:")
    print(f"  末端位置: ({best_gripper_pos[0]:.4f}, {best_gripper_pos[1]:.4f}, {best_gripper_pos[2]:.4f})")
    print(f"  Cube位置:  ({cube_pos[0]:.4f}, {cube_pos[1]:.4f}, {cube_pos[2]:.4f})")
    print(f"  距离: {best_distance*1000:.2f} mm")
    print(f"  水平距离: {np.linalg.norm(best_gripper_pos[:2] - cube_pos[:2])*1000:.2f} mm")
    print(f"  高度差: {(best_gripper_pos[2] - cube_pos[2])*1000:.2f} mm")
    
    can_touch = best_distance < 0.05
    print(f"\n  {'✅ 可以接触！' if can_touch else '❌ 无法接触'}")
    print(f"{'='*80}")
    
    return best_config, best_distance, best_gripper_pos


def visualize_joint_space():
    """可视化关节空间探索"""
    print("\n生成关节空间可视化...")
    
    model = mujoco.MjModel.from_xml_path((get_so100_models_dir() / 'push_cube_loop.xml').as_posix())
    data = mujoco.MjData(model)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    joint_names = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
    cn_names = ["肩部旋转", "肩部俯仰", "肘部", "腕部俯仰", "腕部旋转", "夹爪"]
    
    for i in range(6):
        print(f"  测试关节 {i}: {joint_names[i]}...")
        joint_values, gripper_pos, distances, heights = test_single_joint(i, model, data)
        
        ax = axes[i]
        
        # 双Y轴
        ax2 = ax.twinx()
        
        # 距离曲线
        line1 = ax.plot(np.rad2deg(joint_values), distances * 1000, 
                        'b-', linewidth=2, label='距离Cube')
        ax.axhline(y=50, color='green', linestyle='--', alpha=0.5, label='接触阈值')
        
        # 高度曲线
        line2 = ax2.plot(np.rad2deg(joint_values), heights * 1000, 
                         'r-', linewidth=2, alpha=0.7, label='末端高度')
        ax2.axhline(y=17, color='orange', linestyle='--', alpha=0.5, label='Cube高度')
        
        ax.set_xlabel('关节角度 (度)', fontsize=10)
        ax.set_ylabel('距离 Cube (mm)', color='b', fontsize=10)
        ax2.set_ylabel('末端高度 (mm)', color='r', fontsize=10)
        ax.set_title(f'{i}. {joint_names[i]} ({cn_names[i]})', fontsize=11, fontweight='bold')
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left', fontsize=8)
        
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')
    
    plt.tight_layout()
    joint_space_path = get_outputs_dir() / 'joint_space_analysis.png'
    plt.savefig(joint_space_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 关节空间分析图保存: {joint_space_path}")
    plt.show()


def demo_best_config():
    """使用搜索到的最佳配置演示"""
    print("\n" + "=" * 80)
    print("使用最佳配置推动演示")
    print("=" * 80)
    
    # 搜索最佳配置
    best_config, best_distance, best_gripper_pos = find_best_configuration()
    
    if best_distance > 0.1:  # 超过100mm
        print("\n⚠️  警告: 最佳配置距离仍然较远，演示可能看不到明显接触")
    
    model = mujoco.MjModel.from_xml_path((get_so100_models_dir() / 'push_cube_loop.xml').as_posix())
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    renderer = mujoco.Renderer(model, height=480, width=640)
    
    frames = []
    
    # 动作序列
    actions = [
        (np.array([0, -3.14, 3.14, 0.817, 0, -0.157]), 30, "初始姿态"),
        (best_config, 100, "移动到最佳位置"),
        
        # 在最佳位置附近小幅移动（尝试推动）
        (best_config + np.array([0.1, 0, 0, 0, 0, 0]), 80, "向右推"),
        (best_config + np.array([-0.1, 0, 0, 0, 0, 0]), 80, "向左推"),
        (best_config, 50, "回到最佳位置"),
        
        (np.array([0, -2.5, 2.5, 0.5, 0, -0.157]), 60, "抬起"),
        (np.array([0, -3.14, 3.14, 0.817, 0, -0.157]), 50, "回到初始"),
    ]
    
    print("\n执行动作序列:")
    gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Fixed_Jaw")
    cube_positions = []
    
    for i, (target, steps, desc) in enumerate(actions):
        print(f"  阶段 {i+1}/{len(actions)}: {desc}")
        data.ctrl[:6] = target
        
        for _ in range(steps):
            mujoco.mj_step(model, data)
            cube_positions.append(data.qpos[6:9].copy())
            
            # 侧视角渲染
            cam = mujoco.MjvCamera()
            cam.lookat = np.array([0, 0.08, 0.1])
            cam.distance = 0.5
            cam.azimuth = 0
            cam.elevation = -5
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())
    
    cube_positions = np.array(cube_positions)
    cube_moved = np.linalg.norm(cube_positions[-1] - cube_positions[0])
    
    print("\n结果:")
    print(f"  Cube 移动距离: {cube_moved*1000:.2f} mm")
    print(f"  {'✅ Cube 被推动了！' if cube_moved > 0.01 else '❌ Cube 未明显移动'}")
    
    # 保存视频
    video_path = str(get_outputs_dir() / 'best_config_demo.mp4')
    media.write_video(video_path, frames, fps=30)
    print(f"\n视频保存: {video_path}")
    
    return Video(video_path, embed=True, width=640)


if __name__ == "__main__":
    print("请在 Jupyter Notebook 中运行:")
    print()
    print("from lerobot_sim_lab.control.joint_explorer import show_joint_ranges, visualize_joint_space")
    print("from lerobot_sim_lab.control.joint_explorer import find_best_configuration, demo_best_config")
    print()
    print("# 1. 显示所有关节范围")
    print("show_joint_ranges()")
    print()
    print("# 2. 可视化每个关节的影响")
    print("visualize_joint_space()")
    print()
    print("# 3. 搜索最佳配置")
    print("best_config, distance, gripper_pos = find_best_configuration()")
    print()
    print("# 4. 使用最佳配置演示")
    print("video = demo_best_config()")
    print("video")
