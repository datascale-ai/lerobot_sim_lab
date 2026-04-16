"""
分析 push_cube_loop.xml 场景的详细组件
"""

import matplotlib.pyplot as plt
import mediapy as media
import mujoco
import numpy as np
from IPython.display import Video
from matplotlib.patches import Rectangle

from lerobot_sim_lab.utils.paths import get_outputs_dir, get_so100_models_dir

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

def analyze_scene():
    """详细分析场景组件"""
    print("=" * 70)
    print("push_cube_loop.xml 场景分析")
    print("=" * 70)
    
    # 加载模型
    model = mujoco.MjModel.from_xml_path((get_so100_models_dir() / 'push_cube_loop.xml').as_posix())
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    # 1. 场景组件列表
    print("\n【1. 场景包含的组件】")
    print("-" * 70)
    
    components = {
        "机械臂": "SO-100 6自由度机械臂（从 so100.xml 引入）",
        "cube (方块)": "红色立方体，可以自由移动",
        "goal_region_1": "黄色目标区域（半透明，不碰撞）",
        "goal_region_2": "紫色目标区域（半透明，不碰撞）",
        "轨道围栏": "4面白色墙壁，形成矩形通道",
        "floor": "地面（棋盘格纹理）"
    }
    
    for name, desc in components.items():
        print(f"  ✓ {name}: {desc}")
    
    # 2. Cube 详细信息
    print("\n【2. Cube (方块) 详细信息】")
    print("-" * 70)
    
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube")
    
    cube_pos = data.qpos[6:9]
    cube_quat = data.qpos[9:13]
    cube_mass = model.body_mass[cube_body_id]
    cube_size = model.geom_size[cube_geom_id]
    
    print(f"  初始位置: ({cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f}) 米")
    print(f"  尺寸: {cube_size[0]*2:.3f} × {cube_size[1]*2:.3f} × {cube_size[2]*2:.3f} 米")
    print(f"      = {cube_size[0]*2000:.0f} × {cube_size[1]*2000:.0f} × {cube_size[2]*2000:.0f} 毫米")
    print(f"  质量: {cube_mass:.3f} kg")
    print("  颜色: 红色 (rgba: 0.5, 0, 0, 1)")
    print("  自由度: 7 (3个位置 + 4个四元数旋转)")
    print("  摩擦系数: 1.5")
    print("  ✓ 可以被推动、拾起、旋转")
    
    # 3. 目标区域
    print("\n【3. 目标区域 (Goal Regions)】")
    print("-" * 70)
    
    goals = ["goal_region_1", "goal_region_2"]
    colors = ["黄色", "紫色"]
    
    for goal_name, color in zip(goals, colors):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, goal_name)
        pos = model.geom_pos[geom_id]
        size = model.geom_size[geom_id]
        print(f"  {goal_name} ({color}):")
        print(f"    位置: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        print(f"    尺寸: {size[0]*2:.3f} × {size[1]*2:.3f} × {size[2]*2:.3f} 米")
        print("    特性: 半透明，不参与碰撞（仅作为视觉标记）")
    
    # 4. 轨道/围栏
    print("\n【4. 轨道围栏 (Rails)】")
    print("-" * 70)
    
    walls = ["left_wall", "right_wall", "top_wall", "bottom_wall"]
    wall_info = {}
    
    for wall_name in walls:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, wall_name)
        pos = model.geom_pos[geom_id]
        size = model.geom_size[geom_id]
        wall_info[wall_name] = (pos, size)
        print(f"  {wall_name}:")
        print(f"    位置: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        print(f"    尺寸: {size[0]*2:.3f} × {size[1]*2:.3f} × {size[2]*2:.3f} 米")
    
    # 计算轨道尺寸
    track_width = wall_info["right_wall"][0][0] - wall_info["left_wall"][0][0]
    track_length = wall_info["bottom_wall"][0][1] - wall_info["top_wall"][0][1]
    
    print("\n  轨道内部空间:")
    print(f"    宽度: {track_width:.3f} 米 = {track_width*1000:.0f} 毫米")
    print(f"    长度: {track_length:.3f} 米 = {track_length*1000:.0f} 毫米")
    print("    功能: 限制 cube 只能在这个矩形区域内移动")
    
    # 5. 碰撞配置
    print("\n【5. 碰撞检测】")
    print("-" * 70)
    print("  ✓ Cube 与地面会发生碰撞")
    print("  ✓ Cube 与围栏墙壁会发生碰撞")
    print("  ✓ 机械臂末端执行器可以推动 cube")
    print("  ✓ 机械臂可以用夹爪夹住 cube")
    print("  ✗ 目标区域不参与碰撞（仅视觉标记）")
    
    # 6. 任务说明
    print("\n【6. 任务目标】")
    print("=" * 70)
    print("  这是一个「推方块任务」，NOT「拾起放入容器」")
    print()
    print("  任务描述：")
    print("    1. Cube 初始位置在 goal_region_1 (黄色区域)")
    print("    2. 目标：用机械臂推动 cube 到 goal_region_2 (紫色区域)")
    print("    3. Cube 被限制在轨道内，只能沿着 X 方向移动")
    print("    4. 可以选择：")
    print("       - 推动 (push): 用机械臂末端推动 cube")
    print("       - 拖动 (pull): 用夹爪夹住 cube 拖动")
    print("       - 拍打 (tap): 快速触碰让 cube 滑动")
    print()
    print("  任务难点：")
    print("    • Cube 很轻 (0.05kg)，容易被推飞出轨道")
    print("    • 需要精确控制力度和方向")
    print("    • 轨道很窄，机械臂容易碰到墙壁")
    print("=" * 70)
    
    return model, data, cube_pos, wall_info


def visualize_scene_layout():
    """可视化场景俯视图布局"""
    print("\n生成场景俯视图...")
    
    model = mujoco.MjModel.from_xml_path((get_so100_models_dir() / 'push_cube_loop.xml').as_posix())
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制地面
    ax.add_patch(Rectangle((-0.3, -0.1), 0.6, 0.4, 
                           facecolor='lightgray', edgecolor='black', alpha=0.3))
    
    # 绘制围栏
    walls = {
        "left_wall": (-0.125, 0.08, 0.02, 0.11),
        "right_wall": (0.105, 0.08, 0.02, 0.11),
        "top_wall": (-0.125, 0.08, 0.25, 0.02),
        "bottom_wall": (-0.125, 0.17, 0.25, 0.02)
    }
    
    for wall_name, (x, y, w, h) in walls.items():
        ax.add_patch(Rectangle((x, y), w, h, 
                               facecolor='white', edgecolor='black', linewidth=2))
        ax.text(x + w/2, y + h/2, wall_name.replace('_', '\n'), 
               ha='center', va='center', fontsize=7)
    
    # 绘制目标区域
    ax.add_patch(Rectangle((0.025, 0.09), 0.07, 0.09, 
                           facecolor='yellow', alpha=0.3, edgecolor='orange', linewidth=2))
    ax.text(0.06, 0.135, 'Goal 1\n(黄色)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax.add_patch(Rectangle((-0.095, 0.09), 0.07, 0.09, 
                           facecolor='magenta', alpha=0.3, edgecolor='purple', linewidth=2))
    ax.text(-0.06, 0.135, 'Goal 2\n(紫色)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 绘制 cube
    cube_pos = data.qpos[6:9]
    ax.add_patch(Rectangle((cube_pos[0]-0.015, cube_pos[1]-0.015), 0.03, 0.03,
                           facecolor='red', edgecolor='darkred', linewidth=2))
    ax.text(cube_pos[0], cube_pos[1], 'Cube', ha='center', va='center', 
           fontsize=9, color='white', fontweight='bold')
    
    # 绘制机械臂基座
    ax.add_patch(plt.Circle((0, -0.15), 0.04, facecolor='orange', edgecolor='black', linewidth=2))
    ax.text(0, -0.15, 'Robot\nBase', ha='center', va='center', fontsize=8, fontweight='bold')
    
    # 绘制箭头：任务方向
    ax.annotate('', xy=(-0.06, 0.135), xytext=(0.06, 0.135),
               arrowprops=dict(arrowstyle='->', lw=3, color='blue'))
    ax.text(0, 0.16, '推动方向', ha='center', fontsize=12, color='blue', fontweight='bold')
    
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.2, 0.3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (米)', fontsize=12)
    ax.set_ylabel('Y (米)', fontsize=12)
    ax.set_title('push_cube_loop 场景俯视图布局', fontsize=14, fontweight='bold')
    
    # 添加图例
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc='red', label='Cube (可移动)'),
        plt.Rectangle((0, 0), 1, 1, fc='yellow', alpha=0.3, label='Goal 1 (目标区域)'),
        plt.Rectangle((0, 0), 1, 1, fc='magenta', alpha=0.3, label='Goal 2 (目标区域)'),
        plt.Rectangle((0, 0), 1, 1, fc='white', ec='black', label='围栏 (固定墙壁)'),
        plt.Circle((0, 0), 1, fc='orange', label='机械臂基座'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    scene_layout_path = get_outputs_dir() / 'scene_layout.png'
    plt.savefig(scene_layout_path, dpi=150, bbox_inches='tight')
    print(f"✅ 布局图保存: {scene_layout_path}")
    plt.show()


def demo_push_cube_collision():
    """演示推方块和碰撞"""
    print("\n生成推方块碰撞演示...")
    
    model = mujoco.MjModel.from_xml_path((get_so100_models_dir() / 'push_cube_loop.xml').as_posix())
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    renderer = mujoco.Renderer(model, height=480, width=640)
    
    # 摄像机配置：俯视图
    cam = mujoco.MjvCamera()
    cam.lookat = np.array([0, 0.12, 0])
    cam.distance = 0.5
    cam.azimuth = 90
    cam.elevation = -75
    
    frames = []
    cube_positions = []
    
    # 动作序列：推动 cube 从右到左
    actions = [
        # 阶段1: 初始姿态
        (np.array([0, -3.14, 3.14, 0.817, 0, -0.157]), 30, "初始"),
        
        # 阶段2: 移动到 cube 右侧
        (np.array([0.3, -2.3, 2.5, 0.4, 0, -0.157]), 80, "接近cube"),
        
        # 阶段3: 向左推动
        (np.array([-0.1, -2.1, 2.4, 0.3, 0, -0.157]), 120, "推动中"),
        
        # 阶段4: 继续推到目标
        (np.array([-0.3, -2.3, 2.5, 0.4, 0, -0.157]), 100, "推到目标"),
        
        # 阶段5: 抬起
        (np.array([0, -2.8, 3.0, 0.8, 0, -0.157]), 70, "抬起"),
    ]
    
    print("执行推方块动作...")
    for i, (target, steps, desc) in enumerate(actions):
        print(f"  阶段 {i+1}: {desc}")
        data.ctrl[:6] = target
        
        for step in range(steps):
            mujoco.mj_step(model, data)
            
            # 记录 cube 位置
            cube_pos = data.qpos[6:9].copy()
            cube_positions.append(cube_pos)
            
            # 渲染
            if step % 2 == 0:
                renderer.update_scene(data, camera=cam)
                frames.append(renderer.render())
    
    cube_positions = np.array(cube_positions)
    print(f"✅ 生成 {len(frames)} 帧")
    
    # 绘制 cube 轨迹
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(cube_positions[:, 0], label='X (左右)')
    plt.plot(cube_positions[:, 1], label='Y (前后)')
    plt.axhline(y=0.06, color='orange', linestyle='--', alpha=0.5, label='Goal 1 X')
    plt.axhline(y=-0.06, color='purple', linestyle='--', alpha=0.5, label='Goal 2 X')
    plt.xlabel('步数')
    plt.ylabel('位置 (米)')
    plt.title('Cube 位置轨迹')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.scatter(cube_positions[:, 0], cube_positions[:, 1], 
               c=range(len(cube_positions)), cmap='viridis', s=1)
    plt.colorbar(label='时间步')
    plt.axvline(x=0.06, color='orange', linestyle='--', alpha=0.5, label='Goal 1')
    plt.axvline(x=-0.06, color='purple', linestyle='--', alpha=0.5, label='Goal 2')
    
    # 绘制围栏
    plt.plot([-0.125, -0.125], [0.09, 0.18], 'k-', linewidth=2, label='围栏')
    plt.plot([0.125, 0.125], [0.09, 0.18], 'k-', linewidth=2)
    plt.plot([-0.125, 0.125], [0.09, 0.09], 'k-', linewidth=2)
    plt.plot([-0.125, 0.125], [0.18, 0.18], 'k-', linewidth=2)
    
    plt.xlabel('X 位置 (米)')
    plt.ylabel('Y 位置 (米)')
    plt.title('Cube 轨迹俯视图')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    
    plt.tight_layout()
    cube_trajectory_path = get_outputs_dir() / 'cube_trajectory.png'
    plt.savefig(cube_trajectory_path, dpi=150)
    print(f"✅ 轨迹图保存: {cube_trajectory_path}")
    plt.show()
    
    # 保存视频
    video_path = str(get_outputs_dir() / 'push_cube_demo.mp4')
    media.write_video(video_path, frames, fps=30)
    print(f"✅ 视频保存: {video_path}")
    
    return Video(video_path, embed=True, width=640)


if __name__ == "__main__":
    print("请在 Jupyter Notebook 中运行:")
    print()
    print("from analyze_push_cube_scene import analyze_scene, visualize_scene_layout")
    print("from analyze_push_cube_scene import demo_push_cube_collision")
    print()
    print("# 1. 分析场景组件")
    print("analyze_scene()")
    print()
    print("# 2. 可视化场景布局")
    print("visualize_scene_layout()")
    print()
    print("# 3. 演示推方块（包含碰撞）")
    print("video = demo_push_cube_collision()")
    print("video")
