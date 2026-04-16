#!/usr/bin/env python3
"""
笔抓取多场景实时可视化调参工具（双终端方案）

支持动态设置笔的位置，每个场景独立配置

终端1（此脚本）：显示MuJoCo viewer窗口，实时更新机械臂姿态
终端2（control 模式）：交互式输入命令调整参数

用法：
    终端1: python -m lerobot_sim_lab.tuning.tune_pen_grab_multi --mode live --scenario 1
    终端2: python -m lerobot_sim_lab.tuning.tune_pen_grab_multi --mode control --scenario 1
"""

import argparse
import json
import time
from pathlib import Path

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
    """动态设置所有笔的位置，并固定纸盒位置
    
    Args:
        pens_config: dict, {pen_name: (pos, quat)}
    """
    # 首先固定纸盒位置（防止漂移）
    data.qpos[BOX_QPOS_START:BOX_QPOS_START+3] = BOX_POSITION
    data.qpos[BOX_QPOS_START+3:BOX_QPOS_START+7] = BOX_QUATERNION
    
    # 设置所有笔的位置
    for pen_name, (pos, quat) in pens_config.items():
        if pen_name in PEN_QPOS_MAP:
            qpos_start = PEN_QPOS_MAP[pen_name]
            # 设置位置 (x, y, z)
            data.qpos[qpos_start:qpos_start+3] = pos
            # 设置四元数 (w, x, y, z)
            data.qpos[qpos_start+3:qpos_start+7] = quat
    
    mujoco.mj_forward(model, data)


def main():
    parser = argparse.ArgumentParser(description="笔抓取多场景实时调参工具")
    parser.add_argument('--scenario', type=int, required=True, choices=[0,1,2,3,4,5],
                       help="场景ID (0-5)")
    args = parser.parse_args()
    
    scenario = next(s for s in PEN_SCENARIOS if s['id'] == args.scenario)
    
    # 每个场景独立的目录和配置文件
    scenario_dir = Path(f'pen_grab_tuning/scenario_{scenario["id"]}')
    config_file = scenario_dir / 'live_config.json'
    waypoints_file = scenario_dir / 'waypoints.json'
    scenario_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始配置：home姿态
    home_config = np.array([0, -3.14, 3.14, 0.817, 0, -0.157])
    
    # 初始化配置文件
    config_data = {
        'scenario_id': scenario['id'],
        'scenario_name': scenario['name'],
        'pens': {pen_name: {'pos': pos.tolist(), 'quat': quat} 
                 for pen_name, (pos, quat) in scenario['pens'].items()},
        'current_config': home_config.tolist(),
        'mode': 'manual',
        'updated': time.time(),
    }
    with open(config_file, 'w') as f:
        json.dump(config_data, f)
    
    # 初始化waypoints文件
    if not waypoints_file.exists():
        waypoints_data = {
            'scenario_id': scenario['id'],
            'scenario_name': scenario['name'],
            'pens': {pen_name: {'pos': pos.tolist(), 'quat': quat} 
                     for pen_name, (pos, quat) in scenario['pens'].items()},
            'waypoints': [],
            'description': f'Pen grab trajectory for scenario {scenario["id"]}: {scenario["name"]}'
        }
        with open(waypoints_file, 'w', encoding='utf-8') as f:
            json.dump(waypoints_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print("🖊️  笔抓取多场景调参工具")
    print(f"{'='*80}\n")
    print(f"场景 {scenario['id']}: {scenario['name']}")
    print(f"描述: {scenario['description']}")
    print("笔位置:")
    for pen_name, (pos, quat) in scenario['pens'].items():
        print(f"  {pen_name}: {pos}")
    print("目标: 将所有笔放入纸盒")
    print(f"场景文件: {get_so100_models_dir() / 'scene.xml'}")
    print(f"配置目录: {scenario_dir}")
    print("Viewer窗口已启动")
    print("\n在另一个终端运行:")
    print(f"  python -m lerobot_sim_lab.tuning.tune_pen_grab_multi --mode control --scenario {scenario['id']}")
    print("\n等待控制命令...\n")
    
    # 加载模型
    model = mujoco.MjModel.from_xml_path((get_so100_models_dir() / 'scene.xml').as_posix())
    data = mujoco.MjData(model)
    
    # 重置到keyframe中的home姿态
    home_keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    if home_keyframe_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_keyframe_id)
    else:
        mujoco.mj_resetData(model, data)
        data.qpos[:6] = home_config
    
    # 设置控制器目标位置为home姿态
    data.ctrl[:6] = home_config
    
    # 动态设置所有笔的位置（只做运动学计算，不运行物理引擎）
    set_pens_positions(model, data, scenario['pens'])
    
    print("✅ 所有笔位置已设置:")
    for pen_name, (pos, quat) in scenario['pens'].items():
        print(f"   {pen_name}: {pos}")
    print(f"✅ 机械臂初始姿态 (home): {home_config}")
    print("现在可以在控制端调整参数了\n")
    
    last_mtime = config_file.stat().st_mtime
    last_config = home_config.copy()
    
    print("🔄 开始监控配置文件变化...")
    print(f"📁 监控文件: {config_file}")
    print(f"💾 waypoints保存在: {waypoints_file}")
    print()
    
    # 创建viewer并运行
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 使用 FREE 模式，但从 camera_front_new 获取初始视角
        # camera_front_new: pos="0.28 0.12 1.0" target="Base"
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
        
        # 计算方位角和仰角
        # camera_front_new 大致是从前上方俯视
        viewer.cam.azimuth = -90  # 从前方看
        viewer.cam.elevation = -45  # 俯视45度
        
        print("摄像机视角：camera_front_new 风格 (可用鼠标拖拽调整)\n")
        
        frame_count = 0
        while viewer.is_running():
            # 检查配置文件是否更新
            try:
                current_mtime = config_file.stat().st_mtime
                if current_mtime != last_mtime:
                    with open(config_file) as f:
                        config_data = json.load(f)
                    
                    new_config = np.array(config_data['current_config'])
                    
                    # 如果配置有变化，更新并打印
                    if not np.allclose(new_config, last_config):
                        data.qpos[:6] = new_config
                        data.ctrl[:6] = new_config  # 同时设置控制器目标位置
                        
                        # 重新设置所有笔的位置（确保不被覆盖）
                        set_pens_positions(model, data, scenario['pens'])
                        
                        mujoco.mj_forward(model, data)
                        
                        # 找出变化的关节
                        diff_indices = np.where(~np.isclose(new_config, last_config))[0]
                        print(f"[{time.strftime('%H:%M:%S')}] 配置已更新:")
                        for idx in diff_indices:
                            print(f"  {JOINT_NAMES[idx]:15s}: {new_config[idx]:7.4f} rad ({np.rad2deg(new_config[idx]):7.2f}°)")
                        
                        last_config = new_config.copy()
                    
                    last_mtime = current_mtime
            except Exception as e:
                print(f"读取配置文件出错: {e}")
            
            # 只做运动学计算，不运行物理引擎（与 tune_pen_grab_live.py 一致）
            viewer.sync()
            time.sleep(0.01)  # 100Hz


if __name__ == "__main__":
    main()
