#!/usr/bin/env python3
"""
笔抓取轨迹回放和录制工具

用法：
    python -m lerobot_sim_lab.tuning.tune_pen_grab --mode playback [选项]
    
选项：
    --steps-per-waypoint N    每两个关键帧之间的默认插值步数 (默认: 50)
                              注意: 如果关键帧指定了'steps'参数，会优先使用
    --record                  录制视频
    --output-dir DIR          视频保存目录 (默认: pen_grab_tuning/videos)
    --waypoints START END     只回放第START到第END个关键帧 (1-based索引)
    
示例：
    # 完整回放（包含 waypoints.json 中的所有帧）
    python -m lerobot_sim_lab.tuning.tune_pen_grab --mode playback --record
    
    # 调试特定动作（第2-4帧）
    python -m lerobot_sim_lab.tuning.tune_pen_grab --mode playback --waypoints 2 4
    
    # 调试并录制
    python -m lerobot_sim_lab.tuning.tune_pen_grab --mode playback --waypoints 2 4 --record
    
功能：
    - 读取 pen_grab_tuning/waypoints.json 中的关键帧
    - 支持每个关键帧自定义插值步数（通过'steps'参数）
    - 在关键帧之间生成插值轨迹
    - 回放轨迹（可选录制视频）
    
注意：
    - waypoints.json 的第一帧应该是 home 起始帧（会在保存第一个用户帧时自动添加）
    - 如果使用 --waypoints 截取范围，可能不包含 home 帧

waypoints.json 格式：
    {
      "waypoints": [
        {
          "name": "0-home",       // 起始帧（自动添加）
          "config": [关节1, 关节2, ..., 关节6],
          "steps": 40,
          "timestamp": ...
        },
        {
          "name": "1-第一个动作",
          "config": [关节1, 关节2, ..., 关节6],
          "steps": 50,  // 可选：从前一帧到当前帧的插值步数（25-50推荐）
          "timestamp": ...
        },
        ...
      ]
    }
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import mediapy as media
import mujoco
import mujoco.viewer
import numpy as np

from lerobot_sim_lab.utils.paths import get_so100_models_dir

JOINT_NAMES = ['Rotation', 'Pitch', 'Elbow', 'Wrist_Pitch', 'Wrist_Roll', 'Jaw']


def interpolate_trajectory(waypoints, steps_per_segment=50):
    """
    在关键帧之间生成插值轨迹
    
    Args:
        waypoints: 关键帧列表 [{'name': ..., 'config': [6个关节角度], 'steps': N (可选)}, ...]
        steps_per_segment: 每两个关键帧之间的默认步数（当关键帧未指定steps时使用）
    
    Returns:
        trajectory: 完整轨迹 [(config, description), ...]
    
    说明:
        - 每个关键帧可以指定'steps'参数，表示从前一帧到当前帧的插值步数
        - 如果关键帧未指定'steps'，则使用steps_per_segment作为默认值
    """
    if len(waypoints) < 2:
        print("⚠️  至少需要2个关键帧才能生成轨迹")
        return []
    
    trajectory = []
    
    for i in range(len(waypoints) - 1):
        start_config = np.array(waypoints[i]['config'])
        end_config = np.array(waypoints[i+1]['config'])
        start_name = waypoints[i]['name']
        end_name = waypoints[i+1]['name']
        
        # 获取当前段的步数：使用目标帧（i+1）的steps参数
        # 如果目标帧没有指定steps，则使用默认值
        current_steps = waypoints[i+1].get('steps', steps_per_segment)
        
        # 线性插值
        for step in range(current_steps):
            alpha = step / current_steps
            config = start_config + alpha * (end_config - start_config)
            desc = f"{start_name} → {end_name} ({step}/{current_steps})"
            trajectory.append((config, desc))
    
    # 添加最后一个关键帧
    final_config = np.array(waypoints[-1]['config'])
    trajectory.append((final_config, f"Final: {waypoints[-1]['name']}"))
    
    return trajectory


def playback_trajectory(model, data, trajectory, record=False, output_dir=None):
    """
    回放轨迹并可选录制视频
    
    Args:
        model: MuJoCo模型
        data: MuJoCo数据
        trajectory: 轨迹 [(config, description), ...]
        record: 是否录制视频
        output_dir: 视频保存目录
    
    Returns:
        success: 是否成功回放
    """
    if not trajectory:
        print("❌ 轨迹为空，无法回放")
        return False
    
    print(f"\n{'='*80}")
    print("🎬 开始回放轨迹")
    print(f"{'='*80}")
    print(f"总步数: {len(trajectory)}")
    print(f"录制视频: {'是' if record else '否'}")
    print(f"{'='*80}\n")
    
    frames = []
    
    # 计算substeps：控制频率30Hz，物理时间步0.002s
    # 每个控制步需要运行 (1/30) / 0.002 ≈ 16-17 个物理步骤
    n_substeps = int((1/30) / model.opt.timestep)
    
    # 设置录制摄像机（如果需要）
    if record:
        cam = mujoco.MjvCamera()
        cam.lookat = np.array([0.2, -0.35, 0.80])
        cam.distance = 1.0
        cam.azimuth = 120
        cam.elevation = -30
    
    # 重置场景
    home_keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    if home_keyframe_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_keyframe_id)
    else:
        mujoco.mj_resetData(model, data)
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 设置viewer摄像机视角
        viewer.cam.lookat[:] = [0.2, -0.35, 0.80]
        viewer.cam.distance = 1.0
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -30
        
        print("开始回放...")
        if record:
            print("💡 提示: Viewer显示可能较慢，但录制的视频速度是正常的\n")
        
        for step_idx, (config, desc) in enumerate(trajectory):
            if not viewer.is_running():
                print("\n⚠️  Viewer被关闭，回放中断")
                return False
            
            # 设置关节目标位置
            data.ctrl[:6] = config
            
            # 运行多个物理步骤（让控制器有时间响应）
            for substep in range(n_substeps):
                mujoco.mj_step(model, data)
                
                # 只在最后一个substep更新viewer，减少渲染负担
                # 但仍然保持每个substep都在仿真
                if substep == n_substeps - 1:
                    viewer.sync()
            
            # 录制帧（每个控制步录制一帧，30Hz）
            if record:
                # 渲染当前帧
                renderer = mujoco.Renderer(model, height=480, width=640)
                renderer.update_scene(data, camera=cam)
                frame = renderer.render()
                frames.append(frame)
            
            # 每50步打印一次进度
            if step_idx % 50 == 0:
                print(f"  步骤 {step_idx}/{len(trajectory)}: {desc}")
    
    print("\n✅ 回放完成！")
    
    # 保存视频
    if record and frames:
        if output_dir is None:
            output_dir = Path('pen_grab_tuning/videos')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = output_dir / f"pen_grab_{timestamp}.mp4"
        
        print("\n📹 正在保存视频...")
        media.write_video(str(video_path), frames, fps=30)
        print(f"✅ 视频已保存: {video_path}")
        print(f"   总帧数: {len(frames)}")
        print("   分辨率: 640x480")
        print(f"   时长: {len(frames)/50:.1f}秒")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="笔抓取轨迹回放工具")
    parser.add_argument('--steps-per-waypoint', type=int, default=50,
                       help="每两个关键帧之间的插值步数 (默认: 50)")
    parser.add_argument('--record', action='store_true',
                       help="录制视频")
    parser.add_argument('--output-dir', type=str, default=None,
                       help="视频保存目录 (默认: pen_grab_tuning/videos)")
    parser.add_argument('--waypoints', type=int, nargs=2, metavar=('START', 'END'),
                       help="只回放指定范围的关键帧 (1-based索引，包含两端)，例如: --waypoints 2 4")
    args = parser.parse_args()
    
    waypoints_file = Path('pen_grab_tuning/waypoints.json')
    
    # 检查waypoints文件
    if not waypoints_file.exists():
        print(f"\n❌ 错误: {waypoints_file} 不存在")
        print("请先使用调参工具保存关键帧:")
        print("  终端1: python -m lerobot_sim_lab.tuning.tune_pen_grab --mode live")
        print("  终端2: python -m lerobot_sim_lab.tuning.tune_pen_grab --mode control")
        return
    
    # 加载关键帧
    with open(waypoints_file) as f:
        waypoints_data = json.load(f)
    
    waypoints = waypoints_data['waypoints']
    
    if not waypoints:
        print("\n❌ 错误: waypoints.json 中没有关键帧")
        print("请先使用 python -m lerobot_sim_lab.tuning.tune_pen_grab --mode control 保存关键帧")
        return
    
    # 过滤关键帧范围
    if args.waypoints:
        start_idx, end_idx = args.waypoints
        # 转换为 0-based 索引
        start_idx = max(1, start_idx) - 1
        end_idx = min(len(waypoints), end_idx)
        
        if start_idx >= len(waypoints) or start_idx >= end_idx:
            print(f"\n❌ 错误: 无效的关键帧范围 {args.waypoints[0]}-{args.waypoints[1]}")
            print(f"   可用范围: 1-{len(waypoints)}")
            return
        
        print(f"\n📌 仅回放关键帧 {start_idx+1} 到 {end_idx}")
        original_count = len(waypoints)
        waypoints = waypoints[start_idx:end_idx]
        print(f"   已选择 {len(waypoints)}/{original_count} 个关键帧\n")
    
    # 加载模型（提前加载以获取timestep信息）
    print("📂 加载场景...")
    model = mujoco.MjModel.from_xml_path((get_so100_models_dir() / 'scene.xml').as_posix())
    data = mujoco.MjData(model)
    
    # 计算控制参数
    n_substeps = int((1/30) / model.opt.timestep)
    
    print(f"\n{'='*80}")
    print("🖊️  笔抓取轨迹回放")
    print(f"{'='*80}")
    print(f"场景文件: {get_so100_models_dir() / 'scene.xml'}")
    print(f"关键帧数量: {len(waypoints)}")
    if args.waypoints:
        print(f"回放范围: 关键帧 {args.waypoints[0]} 到 {args.waypoints[1]} (调试模式)")
    print(f"插值步数: {args.steps_per_waypoint} 步/段 (默认值)")
    print("控制频率: 30 Hz")
    print(f"物理时间步: {model.opt.timestep*1000:.1f} ms")
    print(f"物理步/控制步: {n_substeps}")
    print(f"{'='*80}\n")
    
    # 显示关键帧
    print("📍 关键帧列表:")
    for i, wp in enumerate(waypoints, 1):
        config = wp['config']
        steps = wp.get('steps', args.steps_per_waypoint)
        print(f"  {i}. {wp['name']}: [{config[0]:.3f}, {config[1]:.3f}, {config[2]:.3f}, {config[3]:.3f}, {config[4]:.3f}, {config[5]:.3f}] (步数: {steps})")
    print()
    
    # 生成轨迹
    print("🔧 生成插值轨迹...")
    trajectory = interpolate_trajectory(waypoints, args.steps_per_waypoint)
    
    if not trajectory:
        print("❌ 无法生成轨迹")
        return
    
    print(f"✅ 轨迹生成完成，共 {len(trajectory)} 步\n")
    
    # 加载模型
    print("📂 加载场景...")
    model = mujoco.MjModel.from_xml_path((get_so100_models_dir() / 'scene.xml').as_posix())
    data = mujoco.MjData(model)
    print("✅ 场景加载完成\n")
    
    # 回放轨迹
    output_dir = Path(args.output_dir) if args.output_dir else None
    success = playback_trajectory(model, data, trajectory, args.record, output_dir)
    
    if success:
        print(f"\n{'='*80}")
        print("🎉 任务完成！")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
