#!/usr/bin/env python3
"""
笔抓取多场景轨迹回放工具

支持回放指定场景的 waypoints 轨迹，并可选录制视频

用法：
    python -m lerobot_sim_lab.tuning.tune_pen_grab_multi --mode playback --scenario 1
    python -m lerobot_sim_lab.tuning.tune_pen_grab_multi --mode playback --scenario 1 --record
    python -m lerobot_sim_lab.tuning.tune_pen_grab_multi --mode playback --scenario 1 --waypoints 2 4
"""

import argparse
import json
import time
from pathlib import Path

import mediapy
import mujoco
import mujoco.viewer
import numpy as np

from lerobot_sim_lab.config.scenarios.pen_grab import (
    BOX_POSITION,
    BOX_QPOS_START,
    BOX_QUATERNION,
    PEN_QPOS_MAP,
    PEN_SCENARIOS,
)
from lerobot_sim_lab.utils.paths import get_so100_models_dir

JOINT_NAMES = ['Rotation', 'Pitch', 'Elbow', 'Wrist_Pitch', 'Wrist_Roll', 'Jaw']


def set_pens_positions(model, data, pens_config):
    """动态设置所有笔的位置，并固定纸盒位置（用于回放初始化）"""
    # 首先固定纸盒位置（防止漂移）
    data.qpos[BOX_QPOS_START:BOX_QPOS_START+3] = BOX_POSITION
    data.qpos[BOX_QPOS_START+3:BOX_QPOS_START+7] = BOX_QUATERNION
    data.qvel[BOX_QPOS_START:BOX_QPOS_START+6] = 0  # 设置box速度为0
    
    # 设置所有笔的位置和速度
    for pen_name, (pos, quat) in pens_config.items():
        if pen_name in PEN_QPOS_MAP:
            qpos_start = PEN_QPOS_MAP[pen_name]
            # 设置位置
            data.qpos[qpos_start:qpos_start+3] = pos
            data.qpos[qpos_start+3:qpos_start+7] = quat
            # 设置速度为0（freejoint有6个速度分量：3线速度+3角速度）
            data.qvel[qpos_start:qpos_start+6] = 0
    
    mujoco.mj_forward(model, data)


def interpolate_trajectory(waypoints, steps_per_segment=50):
    """
    在关键帧之间生成插值轨迹（与 tune_pen_grab_playback.py 一致）
    
    Args:
        waypoints: 关键帧列表 [{'name': ..., 'config': [6个关节角度], 'steps': N (可选)}, ...]
        steps_per_segment: 每两个关键帧之间的默认步数（当关键帧未指定steps时使用）
    
    Returns:
        trajectory: 完整轨迹，numpy array (N, 6)
    """
    if len(waypoints) < 2:
        print("⚠️  至少需要2个关键帧才能生成轨迹")
        return np.array([])
    
    trajectory = []
    
    for i in range(len(waypoints) - 1):
        start_config = np.array(waypoints[i]['config'])
        end_config = np.array(waypoints[i+1]['config'])
        
        # 获取当前段的步数：使用目标帧（i+1）的steps参数
        current_steps = waypoints[i+1].get('steps', steps_per_segment)
        
        # 线性插值
        for step in range(current_steps):
            alpha = step / current_steps
            config = start_config + alpha * (end_config - start_config)
            trajectory.append(config)
    
    # 添加最后一个关键帧
    final_config = np.array(waypoints[-1]['config'])
    trajectory.append(final_config)
    
    return np.array(trajectory)


def main():
    parser = argparse.ArgumentParser(description="笔抓取多场景轨迹回放工具")
    parser.add_argument('--scenario', type=int, required=True, choices=[0,1,2,3,4,5],
                       help="场景ID (0-5)")
    parser.add_argument('--record', action='store_true',
                       help="录制视频")
    parser.add_argument('--waypoints', nargs=2, type=int, metavar=('START', 'END'),
                       help="只回放指定范围的关键帧 (例如: --waypoints 2 4 回放第2-4帧)")
    args = parser.parse_args()
    
    scenario = next(s for s in PEN_SCENARIOS if s['id'] == args.scenario)
    
    # 场景目录和配置文件
    scenario_dir = Path(f'pen_grab_tuning/scenario_{scenario["id"]}')
    waypoints_file = scenario_dir / 'waypoints.json'
    
    if not waypoints_file.exists():
        print(f"\n❌ 错误: {waypoints_file} 不存在")
        print("请先使用调参工具保存关键帧:")
        print(f"  python -m lerobot_sim_lab.tuning.tune_pen_grab_multi --mode live --scenario {args.scenario}")
        print(f"  python -m lerobot_sim_lab.tuning.tune_pen_grab_multi --mode control --scenario {args.scenario}")
        return
    
    # 加载waypoints
    with open(waypoints_file) as f:
        waypoints_data = json.load(f)
    
    waypoints = waypoints_data['waypoints']
    
    if len(waypoints) == 0:
        print("\n❌ 错误: waypoints文件中没有保存任何关键帧")
        return
    
    print(f"\n{'='*80}")
    print("🖊️  笔抓取多场景轨迹回放")
    print(f"{'='*80}\n")
    print(f"场景 {scenario['id']}: {scenario['name']}")
    print(f"描述: {scenario['description']}")
    print("笔位置:")
    for pen_name, (pos, quat) in scenario['pens'].items():
        print(f"  {pen_name}: {pos}")
    print(f"关键帧文件: {waypoints_file}")
    print(f"关键帧数量: {len(waypoints)}")
    
    # 确定回放范围
    start_idx = 0
    end_idx = len(waypoints) - 1
    
    if args.waypoints:
        start_idx = args.waypoints[0] - 1  # 用户输入是1-based
        end_idx = args.waypoints[1] - 1
        
        if start_idx < 0 or end_idx >= len(waypoints) or start_idx > end_idx:
            print(f"\n❌ 错误: 无效的waypoints范围 [{args.waypoints[0]}, {args.waypoints[1]}]")
            print(f"   有效范围: [1, {len(waypoints)}]")
            return
        
        print(f"回放范围: 第 {start_idx+1}-{end_idx+1} 帧")
        
        # 直接切片waypoints数组（与 tune_pen_grab_playback.py 一致）
        waypoints = waypoints[start_idx:end_idx+1]
    else:
        print(f"回放范围: 全部 {len(waypoints)} 帧")
    
    print("\n关键帧列表:")
    for i, wp in enumerate(waypoints, 1):
        steps = wp.get('steps', 50)
        print(f"  {i}. {wp['name']} (步数: {steps})")
    
    # 生成轨迹（直接传入waypoints，不需要start_idx/end_idx）
    print("\n生成插值轨迹...")
    trajectory = interpolate_trajectory(waypoints, steps_per_segment=50)
    print(f"✅ 轨迹长度: {len(trajectory)} 步")
    
    print(f"\n{'='*80}")
    print("开始回放...")
    print(f"{'='*80}\n")
    
    # 加载模型
    model = mujoco.MjModel.from_xml_path((get_so100_models_dir() / 'scene.xml').as_posix())
    data = mujoco.MjData(model)
    
    # 初始化到home姿态
    home_keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    if home_keyframe_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_keyframe_id)
    else:
        mujoco.mj_resetData(model, data)
    
    # 动态设置所有笔和box的位置（支持多场景）
    set_pens_positions(model, data, scenario['pens'])
    
    # 设置机械臂到起始帧的姿态
    initial_config = np.array(waypoints[0]['config'])
    data.ctrl[:6] = initial_config
    data.qpos[:6] = initial_config
    
    # 只做运动学计算，不运行物理引擎（与 tune_pen_grab_multi_live.py 一致）
    mujoco.mj_forward(model, data)
    
    # 计算控制参数（与 tune_pen_grab_playback.py 一致）
    n_substeps = int((1/30) / model.opt.timestep)
    
    print("✅ 初始化完成")
    print(f"   场景: {scenario['name']}")
    print(f"   机械臂初始姿态: {waypoints[0]['name']}")
    print(f"   控制频率: 30 Hz, 物理步/控制步: {n_substeps}\n")
    
    # 回放轨迹
    frames_front = []
    frames_wrist = []
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 使用 FREE 模式，但从 camera_front_new 获取初始视角（允许鼠标拖拽）
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        
        # 获取Base的位置作为lookat点
        base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'Base')
        if base_body_id >= 0:
            base_pos = data.xpos[base_body_id]
            viewer.cam.lookat[:] = base_pos
        else:
            viewer.cam.lookat[:] = [0.28, 0.08, 0.75]  # Base的大致位置
        
        # 从camera_front_new的位置计算距离和角度
        cam_pos = np.array([0.28, 0.12, 1.0])
        distance = np.linalg.norm(cam_pos - viewer.cam.lookat)
        viewer.cam.distance = distance
        
        # 设置方位角和仰角
        viewer.cam.azimuth = -90  # 从前方看
        viewer.cam.elevation = -45  # 俯视45度
        
        print("开始回放...")
        if args.record:
            print("💡 提示: Viewer显示可能较慢，但录制的视频速度是正常的\n")
        
        for step_idx, config in enumerate(trajectory):
            if not viewer.is_running():
                print("\n⚠️  Viewer被关闭，回放中断")
                return
            
            # 设置关节目标位置
            data.ctrl[:6] = config
            
            # 运行多个物理步骤（让控制器有时间响应，与 tune_pen_grab_playback.py 一致）
            for substep in range(n_substeps):
                mujoco.mj_step(model, data)
                
                # 只在最后一个substep更新viewer，减少渲染负担
                if substep == n_substeps - 1:
                    viewer.sync()
            
            # 录制帧（每个控制步录制一帧，30Hz）
            if args.record:
                # 录制front视角 (480p分辨率，避免超过framebuffer限制)
                renderer_front = mujoco.Renderer(model, height=480, width=640)
                renderer_front.update_scene(data, camera='camera_front_new')
                frame_front = renderer_front.render()
                frames_front.append(frame_front)
                
                # 录制wrist视角
                renderer_wrist = mujoco.Renderer(model, height=480, width=640)
                renderer_wrist.update_scene(data, camera='camera_wrist')
                frame_wrist = renderer_wrist.render()
                frames_wrist.append(frame_wrist)
            
            # 每50步打印一次进度
            if step_idx % 50 == 0:
                print(f"  步骤 {step_idx}/{len(trajectory)}")
    
    print("\n✅ 回放完成！")
    
    if args.record and frames_front:
        print("\n保存视频...")
        
        # 保存front视角
        output_file_front = scenario_dir / f'playback_front_{time.strftime("%Y%m%d_%H%M%S")}.mp4'
        mediapy.write_video(str(output_file_front), frames_front, fps=30)
        print(f"✅ Front视角已保存: {output_file_front}")
        
        # 保存wrist视角
        output_file_wrist = scenario_dir / f'playback_wrist_{time.strftime("%Y%m%d_%H%M%S")}.mp4'
        mediapy.write_video(str(output_file_wrist), frames_wrist, fps=30)
        print(f"✅ Wrist视角已保存: {output_file_wrist}")
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
