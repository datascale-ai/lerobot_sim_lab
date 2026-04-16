#!/usr/bin/env python3
"""
测试相机视角脚本

功能：
1. 读取 waypoints.json 的前 N 帧
2. 生成插值轨迹
3. 录制两个相机视角的视频（front 和 wrist）
4. 保存到 pen_grab_tuning/camera_test/ 目录
5. 可选：合并两个视角为一个视频（左右布局）

用法：
    # 分别保存两个视角
    python3 test_camera_views.py --num-waypoints 34
    
    # 合并为一个视频（左右布局）
    python3 test_camera_views.py --num-waypoints 34 --merge
"""
import sys

import numpy as np
import mujoco
import mediapy as media
from pathlib import Path
import json
from datetime import datetime
import argparse

from lerobot_sim_lab.utils.paths import get_so100_scene_path

JOINT_NAMES = ['Rotation', 'Pitch', 'Elbow', 'Wrist_Pitch', 'Wrist_Roll', 'Jaw']


def interpolate_trajectory(waypoints, steps_per_segment=50):
    """
    在关键帧之间生成插值轨迹
    
    Args:
        waypoints: 关键帧列表 [{'name': ..., 'config': [6个关节角度], 'steps': N}, ...]
        steps_per_segment: 默认步数（当关键帧未指定steps时使用）
    
    Returns:
        trajectory: [(config, description), ...]
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
        
        # 获取当前段的步数
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


def render_dual_view(model, data, renderer_front, renderer_wrist):
    """渲染两个相机视角"""
    # 获取相机ID
    cam_front_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "camera_front_new")
    cam_wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "camera_wrist")
    
    # 渲染 front 视角
    renderer_front.update_scene(data, camera=cam_front_id)
    frame_front = renderer_front.render()
    
    # 渲染 wrist 视角
    renderer_wrist.update_scene(data, camera=cam_wrist_id)
    frame_wrist = renderer_wrist.render()
    
    return frame_front, frame_wrist


def record_trajectory(model_path, trajectory, output_dir, fps=30, merge=False):
    """
    录制轨迹并保存两个相机视角的视频
    
    Args:
        model_path: MuJoCo XML 文件路径
        trajectory: 轨迹 [(config, description), ...]
        output_dir: 输出目录
        fps: 帧率
        merge: 是否合并两个视角为一个视频（左右布局）
    """
    print(f"\n{'='*80}")
    print(f"📹 开始录制相机视角测试")
    print(f"{'='*80}")
    print(f"场景文件: {model_path}")
    print(f"轨迹步数: {len(trajectory)}")
    print(f"录制帧率: {fps} Hz")
    print(f"合并视角: {'是 (左右布局)' if merge else '否 (分别保存)'}")
    print(f"{'='*80}\n")
    
    # 加载模型
    print("📂 加载场景...")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # 创建两个渲染器
    print("🎨 创建渲染器...")
    renderer_front = mujoco.Renderer(model, height=480, width=640)
    renderer_wrist = mujoco.Renderer(model, height=480, width=640)
    
    # 重置到 home 姿态
    home_keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    if home_keyframe_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_keyframe_id)
    else:
        mujoco.mj_resetData(model, data)
    
    # 计算 substeps
    n_substeps = int((1/fps) / model.opt.timestep)
    print(f"⚙️  物理参数: timestep={model.opt.timestep*1000:.1f}ms, substeps={n_substeps}")
    
    frames_front = []
    frames_wrist = []
    
    print(f"\n🎬 开始仿真和录制...")
    for step_idx, (config, desc) in enumerate(trajectory):
        # 设置关节目标位置
        data.ctrl[:6] = config
        
        # 运行多个物理步骤
        for substep in range(n_substeps):
            mujoco.mj_step(model, data)
        
        # 渲染两个视角
        frame_front, frame_wrist = render_dual_view(model, data, renderer_front, renderer_wrist)
        frames_front.append(frame_front)
        frames_wrist.append(frame_wrist)
        
        # 打印进度
        if step_idx % 50 == 0 or step_idx == len(trajectory) - 1:
            print(f"  步骤 {step_idx}/{len(trajectory)}: {desc}")
    
    print(f"\n✅ 仿真完成！共录制 {len(frames_front)} 帧")
    
    # 保存视频
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if merge:
        # 合并两个视角为一个视频（左右布局）
        print(f"\n🔗 合并两个视角...")
        merged_frames = []
        for frame_front, frame_wrist in zip(frames_front, frames_wrist):
            # 左右拼接：front 在左，wrist 在右
            merged_frame = np.hstack([frame_front, frame_wrist])
            merged_frames.append(merged_frame)
        
        video_merged_path = output_dir / f"camera_merged_{timestamp}.mp4"
        print(f"\n💾 保存合并视频: {video_merged_path}")
        media.write_video(str(video_merged_path), merged_frames, fps=fps)
        
        # 修正文件权限
        try:
            import os
            os.chmod(video_merged_path, 0o644)
            import pwd
            uid = pwd.getpwnam('qjp').pw_uid
            gid = pwd.getpwnam('qjp').pw_gid
            os.chown(video_merged_path, uid, gid)
        except (PermissionError, KeyError):
            pass
        
        print(f"\n{'='*80}")
        print(f"🎉 录制完成！")
        print(f"{'='*80}")
        print(f"合并视频: {video_merged_path}")
        print(f"视频时长: {len(merged_frames)/fps:.1f}秒")
        print(f"分辨率: {merged_frames[0].shape[1]}x{merged_frames[0].shape[0]} (1280x480)")
        print(f"{'='*80}\n")
    else:
        # 分别保存两个视角
        # 保存 front 视角
        video_front_path = output_dir / f"camera_front_{timestamp}.mp4"
        print(f"\n💾 保存 Front 视角视频: {video_front_path}")
        media.write_video(str(video_front_path), frames_front, fps=fps)
        
        # 保存 wrist 视角
        video_wrist_path = output_dir / f"camera_wrist_{timestamp}.mp4"
        print(f"💾 保存 Wrist 视角视频: {video_wrist_path}")
        media.write_video(str(video_wrist_path), frames_wrist, fps=fps)
        
        # 修正文件权限（如果以 root 运行）
        import os
        try:
            os.chmod(video_front_path, 0o644)
            os.chmod(video_wrist_path, 0o644)
            # 如果需要，也可以尝试修改所有者（需要 root 权限）
            import pwd
            uid = pwd.getpwnam('qjp').pw_uid
            gid = pwd.getpwnam('qjp').pw_gid
            os.chown(video_front_path, uid, gid)
            os.chown(video_wrist_path, uid, gid)
        except (PermissionError, KeyError):
            # 如果不是 root 或用户不存在，跳过
            pass
        
        print(f"\n{'='*80}")
        print(f"🎉 录制完成！")
        print(f"{'='*80}")
        print(f"Front 视角: {video_front_path}")
        print(f"Wrist 视角: {video_wrist_path}")
        print(f"视频时长: {len(frames_front)/fps:.1f}秒")
        print(f"分辨率: 640x480")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="测试相机视角")
    parser.add_argument('--waypoints-file', type=str, default='pen_grab_tuning/waypoints.json',
                       help="关键帧文件路径")
    parser.add_argument('--num-waypoints', type=int, default=9,
                       help="使用前N个关键帧（默认: 9）")
    parser.add_argument('--scene-xml', type=str, default=str(get_so100_scene_path("scene")),
                       help="场景XML文件路径")
    parser.add_argument('--output-dir', type=str, default='pen_grab_tuning/camera_test',
                       help="视频输出目录")
    parser.add_argument('--fps', type=int, default=30,
                       help="录制帧率（默认: 30）")
    parser.add_argument('--merge', action='store_true',
                       help="合并两个视角为一个视频（左右布局）")
    args = parser.parse_args()
    
    # 加载关键帧
    waypoints_file = Path(args.waypoints_file)
    if not waypoints_file.exists():
        print(f"\n❌ 错误: {waypoints_file} 不存在")
        return
    
    print(f"\n📖 读取关键帧文件: {waypoints_file}")
    with open(waypoints_file, 'r', encoding='utf-8') as f:
        waypoints_data = json.load(f)
    
    all_waypoints = waypoints_data['waypoints']
    
    # 只取前N个关键帧
    waypoints = all_waypoints[:args.num_waypoints]
    
    print(f"✅ 加载了 {len(waypoints)}/{len(all_waypoints)} 个关键帧")
    print(f"\n📍 将使用的关键帧:")
    for i, wp in enumerate(waypoints, 1):
        steps = wp.get('steps', 50)
        print(f"  {i}. {wp['name']} (步数: {steps})")
    
    # 生成轨迹
    print(f"\n🔧 生成插值轨迹...")
    trajectory = interpolate_trajectory(waypoints)
    
    if not trajectory:
        print("❌ 无法生成轨迹")
        return
    
    print(f"✅ 轨迹生成完成，共 {len(trajectory)} 步")
    print(f"   预计视频时长: {len(trajectory)/args.fps:.1f}秒")
    
    # 录制视频
    record_trajectory(args.scene_xml, trajectory, args.output_dir, args.fps, args.merge)


if __name__ == "__main__":
    main()
