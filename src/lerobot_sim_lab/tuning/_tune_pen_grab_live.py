#!/usr/bin/env python3
"""
实时可视化笔抓取调参工具（双终端方案）

终端1（此脚本）：显示MuJoCo viewer窗口，实时更新机械臂姿态
终端2（control 模式）：交互式输入命令调整参数

用法：
    终端1: python -m lerobot_sim_lab.tuning.tune_pen_grab --mode live
    终端2: python -m lerobot_sim_lab.tuning.tune_pen_grab --mode control
"""

import json
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from lerobot_sim_lab.utils.paths import get_so100_models_dir

JOINT_NAMES = ['Rotation', 'Pitch', 'Elbow', 'Wrist_Pitch', 'Wrist_Roll', 'Jaw']

def main():
    # 创建共享文件用于进程间通信
    config_file = Path('pen_grab_tuning/live_config.json')
    waypoints_file = Path('pen_grab_tuning/waypoints.json')
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 初始配置：home姿态
    home_config = np.array([0, -3.14, 3.14, 0.817, 0, -0.157])
    
    # 初始化配置文件
    config_data = {
        'current_config': home_config.tolist(),
        'mode': 'manual',  # 'manual' or 'playback'
        'updated': time.time(),
    }
    with open(config_file, 'w') as f:
        json.dump(config_data, f)
    
    # 初始化waypoints文件
    if not waypoints_file.exists():
        waypoints_data = {
            'waypoints': [],  # 存储关键帧列表
            'description': 'Pen4 grab and place trajectory'
        }
        with open(waypoints_file, 'w') as f:
            json.dump(waypoints_data, f, indent=2)
    
    print(f"\n{'='*80}")
    print("📝 笔抓取任务调参工具")
    print(f"{'='*80}\n")
    print("目标: 抓取 pen_4 并放入纸盒")
    print(f"场景文件: {get_so100_models_dir() / 'scene.xml'}")
    print("Viewer窗口已启动")
    print("\n在另一个终端运行:")
    print("  python -m lerobot_sim_lab.tuning.tune_pen_grab --mode control")
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
    
    mujoco.mj_forward(model, data)
    
    print(f"机械臂初始姿态 (home): {home_config}")
    print("现在可以在控制端调整参数了\n")
    
    last_mtime = config_file.stat().st_mtime
    last_config = home_config.copy()
    
    print("🔄 开始监控配置文件变化...")
    print(f"📁 监控文件: {config_file}")
    print(f"💾 waypoints保存在: {waypoints_file}")
    print()
    
    # 创建viewer并运行
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 设置初始摄像机视角：俯视工作区域
        viewer.cam.lookat[:] = [0.2, -0.35, 0.80]  # 看向笔和纸盒区域
        viewer.cam.distance = 1.0                   # 距离
        viewer.cam.azimuth = 120                    # 侧视角
        viewer.cam.elevation = -30                  # 俯视角度
        print("摄像机视角：俯视工作区域 (azimuth=120°, elevation=-30°)\n")
        
        frame_count = 0
        while viewer.is_running():
            # 检查配置文件是否更新
            try:
                current_mtime = config_file.stat().st_mtime
                if current_mtime != last_mtime:
                    last_mtime = current_mtime
                    
                    # 读取新配置
                    with open(config_file) as f:
                        config_data = json.load(f)
                    
                    current_config = np.array(config_data['current_config'])
                    
                    # 检查配置是否真的变化了
                    if not np.allclose(current_config, last_config):
                        if config_data['mode'] == 'manual':
                            # 手动模式：直接显示姿态
                            data.qpos[:6] = current_config
                            data.ctrl[:6] = current_config
                            mujoco.mj_forward(model, data)
                            
                            print(f"✅ 配置已更新 (frame {frame_count})")
                            for i, (name, val) in enumerate(zip(JOINT_NAMES, current_config)):
                                print(f"   {name:12s}: {val:7.4f}")
                            print()
                            
                            last_config = current_config.copy()
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"❌ 更新配置时出错: {e}")
            
            viewer.sync()
            frame_count += 1
            time.sleep(0.01)  # 100Hz
    
    print("\nViewer窗口已关闭")


if __name__ == "__main__":
    main()
