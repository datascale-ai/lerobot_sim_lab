#!/usr/bin/env python3
"""
实时可视化场景微调工具（双终端方案）

终端1（此脚本）：显示MuJoCo viewer窗口，实时更新机械臂姿态
终端2（tune_control.py）：交互式输入命令调整参数

用法：
    终端1: python3 tune_scenario_live.py --scenario 1
    终端2: python3 tune_control.py --scenario 1
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from lerobot_sim_lab.utils.paths import get_so100_scene_path

SCENARIOS = [
    {
        'id': 1,
        'name': 'Center (0.06, 0.135)',
        'cube_pos': np.array([0.0600, 0.1350, 0.0170]),
        # ✅ 来自硬编码序列中验证成功的抓取姿态（准备抓取，张开夹爪）
        'pick_config': np.array([-0.25, -1.35, 2.2, 0.5, 0, 0.65]),
    },
    {
        'id': 2,
        'name': 'Right +3cm (0.09, 0.135)',
        'cube_pos': np.array([0.0900, 0.1350, 0.0170]),
        # ✅ 通过GUI微调验证成功的配置
        'pick_config': np.array([-0.3500, -1.5800, 2.1100, 0.6060, 0.0000, 0.4300]),
    },
    {
        'id': 3,
        'name': 'Left -2cm (0.02, 0.135)',
        'cube_pos': np.array([0.04, 0.1350, 0.0170]),
        'pick_config': np.array([-0.17, -1.6, 2.24, 0.5, 0, 0.65]),
    },
    {
        'id': 4,
        'name': 'Back +2cm (0.06, 0.155)',
        'cube_pos': np.array([0.0600, 0.1550, 0.0170]),
        'pick_config': np.array([-0.2000, -1.5250, 2.1000, 0.5200, 0.0000, 0.3500]),
    },
    {
        'id': 5,
        'name': 'Front -2cm (0.06, 0.115)',
        'cube_pos': np.array([0.0600, 0.1150, 0.0170]),
        'pick_config': np.array([-0.2500, -1.7100, 2.2050, 0.7450, 0.0000, 0.3500]),
    },
]


def main():
    parser = argparse.ArgumentParser(description="实时可视化调参工具")
    parser.add_argument('--scenario', type=int, required=True, choices=[1,2,3,4,5],
                       help="场景ID (1-5)")
    args = parser.parse_args()
    
    scenario = next(s for s in SCENARIOS if s['id'] == args.scenario)
    
    # 创建共享文件用于进程间通信
    config_file = Path(f'scenario_tuning/scenario_{scenario["id"]}/live_config.json')
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 初始化配置文件
    config_data = {
        'pick_config': scenario['pick_config'].tolist(),
        'mode': 'manual',  # 'manual' or 'test'
        'updated': time.time(),
    }
    with open(config_file, 'w') as f:
        json.dump(config_data, f)
    
    print(f"\n{'='*80}")
    print(f"场景 {scenario['id']}: {scenario['name']}")
    print(f"{'='*80}\n")
    print("Viewer窗口已启动")
    print("\n在另一个终端运行:")
    print(f"  python3 tune_control.py --scenario {args.scenario}")
    print("\n等待控制命令...\n")
    
    # 加载模型
    model = mujoco.MjModel.from_xml_path(str(get_so100_scene_path("push_cube")))
    data = mujoco.MjData(model)
    
    # 设置场景
    mujoco.mj_resetData(model, data)
    data.qpos[6:9] = scenario['cube_pos']
    data.qpos[9:13] = [1, 0, 0, 0]
    # ⭐ 初始就显示pick_config姿态，而不是home姿态
    data.qpos[:6] = scenario['pick_config']
    mujoco.mj_forward(model, data)
    
    print(f"机械臂已移动到抓取姿态: {scenario['pick_config']}")
    print("现在可以在控制端调整参数了\n")
    
    last_mtime = config_file.stat().st_mtime
    last_config = scenario['pick_config'].copy()
    
    print("🔄 开始监控配置文件变化...")
    print(f"📁 监控文件: {config_file}")
    print()
    
    # 创建viewer并运行
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 设置初始摄像机视角：正侧俯视图
        viewer.cam.lookat[:] = [0, 0.10, 0.10]  # 看向机械臂工作区域中心
        viewer.cam.distance = 0.8               # 距离稍远一些以获得全景
        viewer.cam.azimuth = 45                 # 正侧视角（45度，介于正面和侧面之间）
        viewer.cam.elevation = -35              # 俯视角度（-35度，从上往下看）
        print("摄像机视角：正侧俯视图 (azimuth=45°, elevation=-35°)\n")
        
        frame_count = 0
        while viewer.is_running():
            # 检查配置文件是否更新（高频检测）
            try:
                current_mtime = config_file.stat().st_mtime
                if current_mtime != last_mtime:
                    last_mtime = current_mtime
                    
                    # 读取新配置
                    with open(config_file) as f:
                        config_data = json.load(f)
                    
                    pick_config = np.array(config_data['pick_config'])
                    
                    # 检查配置是否真的变化了
                    if not np.allclose(pick_config, last_config):
                        if config_data['mode'] == 'manual':
                            # 手动模式：直接显示姿态
                            data.qpos[:6] = pick_config
                            # 确保物理状态一致
                            data.ctrl[:6] = pick_config
                            mujoco.mj_forward(model, data)
                            
                            print(f"✅ 配置已更新 (frame {frame_count})")
                            print(f"   关节值: [{', '.join(f'{x:.4f}' for x in pick_config)}]")
                            print()
                            
                            last_config = pick_config.copy()
                        elif config_data['mode'] == 'test':
                            # 测试模式：运行完整轨迹
                            print("开始测试轨迹...")
                            # 这里可以添加轨迹测试逻辑
            except FileNotFoundError:
                print(f"⚠️  配置文件不存在: {config_file}")
            except json.JSONDecodeError as e:
                print(f"⚠️  JSON解析错误: {e}")
            except Exception as e:
                print(f"❌ 更新配置时出错: {e}")
            
            viewer.sync()
            frame_count += 1
            time.sleep(0.01)  # 提高检测频率 (100Hz)
    
    print("\nViewer窗口已关闭")


if __name__ == "__main__":
    main()
