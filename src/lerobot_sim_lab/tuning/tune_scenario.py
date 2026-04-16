#!/usr/bin/env python3
"""
单场景微调工具

用法：
    python -m lerobot_sim_lab.tuning.tune_scenario --mode control --scenario 1
    python -m lerobot_sim_lab.tuning.tune_scenario --mode live --scenario 1
    
然后按照提示逐步调整关节角度，直到找到成功的抓取配置
"""
import argparse
import json
import sys
from pathlib import Path

import mediapy as media
import mujoco
import numpy as np

from lerobot_sim_lab.utils.paths import get_so100_scene_path

# 场景定义（固定的cube位置）
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
        'name': 'Left -2cm (0.04, 0.135)',
        'cube_pos': np.array([0.0400, 0.1350, 0.0170]),
        # 待微调：基于场景1调整 Rotation (需要更正，向左转)
        'pick_config': np.array([-0.17, -1.6, 2.24, 0.5, 0, 0.65]),
    },
    {
        'id': 4,
        'name': 'Back +2cm (0.06, 0.155)',
        'cube_pos': np.array([0.0600, 0.1550, 0.0170]),
        # 待微调：cube更远，可能需要调整 Pitch/Elbow
        'pick_config': np.array([-0.2000, -1.5250, 2.1000, 0.5200, 0.0000, 0.3500]),
    },
    {
        'id': 5,
        'name': 'Front -2cm (0.06, 0.115)',
        'cube_pos': np.array([0.0600, 0.1150, 0.0170]),
        # 待微调：cube更近，可能需要调整 Pitch/Elbow
        'pick_config': np.array([-0.2500, -1.7100, 2.2050, 0.7450, 0.0000, 0.3500]),
    },
]

JOINT_NAMES = ['Rotation', 'Pitch', 'Elbow', 'Wrist_Pitch', 'Wrist_Roll', 'Jaw']


def generate_action_sequence(scenario_id):
    """
    根据场景ID返回对应的动作序列
    
    Args:
        scenario_id: 场景编号 (1-5)
    
    Returns:
        list: 动作序列，格式为 [(关节角度, 步数, 描述), ...]
    """
    if scenario_id == 1:
        # === 场景1: Center (0.06, 0.135) - 基准场景 ===
        # 步数优化至10-30范围，总步数: 174 → 340
        return [
            (np.array([0, -3.14, 3.14, 0.817, 0, -0.157]), 15, "初始姿态"),
            (np.array([-0.25, -2.5, 2.5, 0.6, 0, -0.157]), 30, "转向黄色区域"),
            (np.array([-0.25, -1.6, 2.3, 0.5, 0, 0.5]), 30, "下降中..."),
            (np.array([-0.25, -1.45, 2.2, 0.5, 0, 0.6]), 15, "继续下降..."),
            (np.array([-0.25, -1.35, 2.2, 0.5, 0, 0.65]), 15, "张开夹爪"),
            (np.array([-0.25, -1.25, 2.2, 0.5, 0, 0.1]), 30, "闭合夹爪 - 抓取！"),
            (np.array([-0.25, -1.75, 2.1, 0.5, 0, 0.1]), 25, "抬起cube"),
            (np.array([-0.25, -2.0, 2.1, 0.5, 0, 0.1]), 15, "继续抬高"),
            (np.array([0.0, -2.0, 2.2, 0.5, 0, 0.2]), 15, "移动到中央"),
            (np.array([0.25, -1.8, 2.3, 0.5, 0, 0.2]), 20, "移动到紫色区域上方"),
            (np.array([0.25, -1.8, 2.35, 0.5, 0, 0.6]), 30, "张开夹爪 - 释放！"),
            (np.array([0.25, -1.5, 2.0, 0.6, 0, 1.5]), 30, "后退"),
            (np.array([0.25, -2.5, 2.5, 0.7, 0, -0.157]), 30, "抬起"),
            (np.array([0, -2.8, 2.8, 0.7, 0, -0.157]), 20, "移动到中央"),
            (np.array([0, -3.14, 3.14, 0.817, 0, -0.157]), 20, "回到初始姿态"),
        ]
    
    elif scenario_id == 2:
        # === 场景2: Right +3cm (0.09, 0.135) ===
        # 步数优化至10-30范围，总步数: 169 → 330
        return [
            (np.array([0, -3.14, 3.14, 0.817, 0, -0.157]), 15, "初始姿态"),
            (np.array([-0.3500, -2.5, 2.5, 0.6, 0, -0.157]), 30, "转向黄色区域"),
            (np.array([-0.3500, -1.75, 2.3, 0.5, 0, 0.5]), 30, "下降中..."),
            (np.array([-0.3500, -1.5800, 2.200, 0.50, 0, 0.4500]), 15, "继续下降..."),
            (np.array([-0.3500, -1.5800, 2.1100, 0.6060, 0.0000, 0.4300]), 15, "张开夹爪"),
            (np.array([-0.3500, -1.5800, 2.1100, 0.6060, 0.0000, 0.1]), 30, "闭合夹爪 - 抓取！"),
            (np.array([-0.3500, -1.75, 2.1, 0.5, 0, 0.1]), 15, "抬起cube"),
            (np.array([-0.3500, -2.0, 2.1, 0.5, 0, 0.1]), 15, "继续抬高"),
            (np.array([0.0, -2.0, 2.2, 0.5, 0, 0.2]), 20, "移动到中央"),
            (np.array([0.3, -1.8, 2.3, 0.6, 0, 0.2]), 20, "移动到紫色区域上方"),
            (np.array([0.3, -1.8, 2.35, 0.5, 0, 0.6]), 30, "张开夹爪 - 释放！"),
            (np.array([0.3, -1.5, 2.0, 0.6, 0, 0.5]), 20, "后退"),
            (np.array([0.3, -2.5, 2.5, 0.7, 0, -0.157]), 30, "抬起"),
            (np.array([0, -2.8, 2.8, 0.7, 0, -0.157]), 25, "移动到中央"),
            (np.array([0, -3.14, 3.14, 0.817, 0, -0.157]), 20, "回到初始姿态"),
        ]
    elif scenario_id == 3:
        # === 场景3: Left -2cm (0.04, 0.135) ===
        # 步数优化至10-30范围，总步数: 164 → 320
        return [
            (np.array([0, -3.14, 3.14, 0.817, 0, -0.157]), 15, "初始姿态"),
            (np.array([-0.17, -2.0, 2.5, 0.6, 0, -0.157]), 30, "转向黄色区域"),
            (np.array([-0.17, -1.75, 2.3, 0.5, 0, 0.45]), 25, "下降中..."),
            (np.array([-0.17, -1.5800, 2.200, 0.50, 0, 0.500]), 15, "继续下降..."),
            (np.array([-0.17, -1.6, 2.24, 0.5, 0, 0.65]), 15, "张开夹爪"),
            (np.array([-0.17, -1.6, 2.24, 0.5, 0, 0.2]), 30, "闭合夹爪 - 抓取！"),
            (np.array([-0.1, -1.75, 2.1, 0.5, 0, 0.1]), 15, "抬起cube"),
            (np.array([-0.05, -2.0, 2.1, 0.5, 0, 0.1]), 15, "继续抬高"),
            (np.array([0.0, -2.0, 2.2, 0.5, 0, 0.2]), 15, "移动到中央"),
            (np.array([0.3, -1.8, 2.3, 0.6, 0, 0.2]), 20, "移动到紫色区域上方"),
            (np.array([0.3, -1.8, 2.3, 0.6, 0, 0.5]), 30, "张开夹爪 - 释放！"),
            (np.array([0.3, -1.5, 2.0, 0.6, 0, 0.5]), 20, "后退"),
            (np.array([0.3, -2.5, 2.5, 0.7, 0, -0.157]), 30, "抬起"),
            (np.array([0, -2.8, 2.8, 0.7, 0, -0.157]), 25, "移动到中央"),
            (np.array([0, -3.14, 3.14, 0.817, 0, -0.157]), 20, "回到初始姿态"),
        ]
    elif scenario_id == 4:
        # === 场景4: Back +2cm (0.06, 0.155) ===
        # 步数优化至10-30范围，总步数: 162 → 315
        return [
            (np.array([0, -3.14, 3.14, 0.817, 0, -0.157]), 15, "初始姿态"),
            (np.array([-0.220, -2.0, 2.5, 0.6, 0, -0.157]), 30, "转向黄色区域"),
            (np.array([-0.23, -1.75, 2.3, 0.5, 0, 0.2]), 20, "下降中..."),
            (np.array([-0.250, -1.75, 2.2000, 0.5550, 0.0000, 0.3500]), 15, "继续下降..."),
            (np.array([-0.20, -1.5050, 2.1000, 0.5200, 0.0000, 0.400]), 15, "张开夹爪"),
            (np.array([-0.200, -1.5050, 2.1000, 0.5200, 0.0000, 0.200]), 30, "闭合夹爪 - 抓取！"),
            (np.array([-0.2, -1.75, 2.1, 0.5, 0, 0.2]), 15, "抬起cube"),
            (np.array([-0.2, -2.0, 2.1, 0.5, 0, 0.2]), 15, "继续抬高"),
            (np.array([0.0, -2.0, 2.2, 0.5, 0, 0.2]), 15, "移动到中央"),
            (np.array([0.3, -1.8, 2.3, 0.6, 0, 0.2]), 20, "移动到紫色区域上方"),
            (np.array([0.3, -1.8, 2.3, 0.6, 0, 0.5]), 30, "张开夹爪 - 释放！"),
            (np.array([0.3, -1.5, 2.0, 0.6, 0, 0.5]), 20, "后退"),
            (np.array([0.3, -2.5, 2.5, 0.7, 0, -0.157]), 30, "抬起"),
            (np.array([0, -2.8, 2.8, 0.7, 0, -0.157]), 25, "移动到中央"),
            (np.array([0, -3.14, 3.14, 0.817, 0, -0.157]), 20, "回到初始姿态"),
        ]
    elif scenario_id == 5:
        # === 场景5: Front -2cm (0.06, 0.115) ===
        # 步数优化至10-30范围，总步数: 153 → 300
        return [
            (np.array([0, -3.14, 3.14, 0.817, 0, -0.157]), 15, "初始姿态"),
            (np.array([-0.250, -2.4, 2.7, 0.6, 0, -0.157]), 30, "转向黄色区域"),
            (np.array([-0.25, -2, 2.4, 0.7450, 0, 0.2]), 25, "下降中..."),
            (np.array([-0.250, -1.76, 2.3050, 0.6350, 0.0000, 0.3500]), 20, "继续下降..."),
            (np.array([-0.2500, -1.760, 2.3050, 0.6350, 0.0000, 0.20]), 15, "张开夹爪"),
            (np.array([-0.2, -1.75, 2.1, 0.7450, 0, 0.2]), 15, "抬起cube"),
            (np.array([-0.2, -2.0, 2.1, 0.5, 0, 0.2]), 20, "继续抬高"),
            (np.array([0.0, -2.0, 2.2, 0.5, 0, 0.2]), 15, "移动到中央"),
            (np.array([0.3, -1.8, 2.3, 0.6, 0, 0.2]), 20, "移动到紫色区域上方"),
            (np.array([0.3, -1.8, 2.3, 0.6, 0, 0.35]), 30, "张开夹爪 - 释放！"),
            (np.array([0.3, -1.5, 2.0, 0.6, 0, 0.35]), 20, "后退"),
            (np.array([0.3, -2.5, 2.5, 0.7, 0, -0.157]), 30, "抬起"),
            (np.array([0, -2.8, 2.8, 0.7, 0, -0.157]), 25, "移动到中央"),
            (np.array([0, -3.14, 3.14, 0.817, 0, -0.157]), 20, "回到初始姿态"),
        ]
    else:
        raise NotImplementedError(f"场景 {scenario_id} 的动作序列尚未定义，需要手动调试后添加")


def test_configuration(scenario, output_path=None, verbose=True):
    """测试指定配置并返回结果"""
    if verbose:
        print(f"\n{'='*80}")
        print(f"测试场景 {scenario['id']}: {scenario['name']}")
        print(f"Cube位置: {scenario['cube_pos']}")
        print(f"{'='*80}\n")
    
    model = mujoco.MjModel.from_xml_path(str(get_so100_scene_path("push_cube")))
    data = mujoco.MjData(model)
    
    mujoco.mj_resetData(model, data)
    data.qpos[6:9] = scenario['cube_pos']
    data.qpos[9:13] = [1, 0, 0, 0]
    data.qpos[:6] = np.array([0, -3.14, 3.14, 0.817, 0, -0.157])
    mujoco.mj_forward(model, data)
    
    renderer = mujoco.Renderer(model, height=480, width=640)
    action_sequence = generate_action_sequence(scenario['id'])
    
    frames_front = []
    frames_side = []
    cube_positions = []
    
    n_substeps = int((1/30) / 0.002)
    gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Fixed_Jaw")
    
    for action_idx, (target_qpos, steps_in_action, description) in enumerate(action_sequence):
        if verbose:
            print(f"  {action_idx+1:2d}. {description:25s} ({steps_in_action:3d}步)")
        
        start_qpos = data.qpos[:6].copy()
        
        for step in range(steps_in_action):
            alpha = (step + 1) / steps_in_action
            interpolated_qpos = (1.0 - alpha) * start_qpos + alpha * target_qpos
            data.ctrl[:6] = interpolated_qpos
            
            for _ in range(n_substeps):
                mujoco.mj_step(model, data)
            
            cube_positions.append(data.qpos[6:9].copy())
            
            # 渲染
            cam_side = mujoco.MjvCamera()
            cam_side.lookat = np.array([0, 0.08, 0.15])
            cam_side.distance = 0.6
            cam_side.azimuth = 0
            cam_side.elevation = -10
            renderer.update_scene(data, camera=cam_side)
            frames_side.append(renderer.render())
            
            cam_front = mujoco.MjvCamera()
            cam_front.lookat = np.array([0, 0.08, 0.15])
            cam_front.distance = 0.6
            cam_front.azimuth = 270
            cam_front.elevation = -10
            renderer.update_scene(data, camera=cam_front)
            frames_front.append(renderer.render())
    
    cube_positions = np.array(cube_positions)
    start_pos = cube_positions[0]
    end_pos = cube_positions[-1]
    max_height = np.max(cube_positions[:, 2])
    
    lift_amount = (max_height - start_pos[2]) * 1000
    movement_distance = np.linalg.norm(end_pos[:2] - start_pos[:2]) * 1000
    was_lifted = lift_amount > 20
    
    if verbose:
        print("\n  结果:")
        print(f"    抬起高度: {lift_amount:.1f} mm")
        print(f"    移动距离: {movement_distance:.1f} mm")
        print(f"    是否抬起: {'✅ 是' if was_lifted else '❌ 否'}")
    
    if output_path:
        output_path = Path(output_path)
        media.write_video(str(output_path / 'test_front.mp4'), frames_front, fps=30)
        media.write_video(str(output_path / 'test_side.mp4'), frames_side, fps=30)
        if verbose:
            print(f"    视频已保存到: {output_path}")
    
    return {
        'lift_amount_mm': float(lift_amount),
        'movement_distance_mm': float(movement_distance),
        'was_lifted': bool(was_lifted),
        'start_pos': start_pos.tolist(),
        'end_pos': end_pos.tolist(),
        'max_height': float(max_height),
    }


def interactive_tuning(scenario):
    """交互式微调"""
    print(f"\n{'='*80}")
    print(f"场景 {scenario['id']}: {scenario['name']}")
    print(f"Cube 位置: {scenario['cube_pos']}")
    print(f"{'='*80}\n")
    
    current_config = scenario['pick_config'].copy()
    output_dir = Path(f'scenario_tuning/scenario_{scenario["id"]}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("当前抓取配置:")
    for i, (name, value) in enumerate(zip(JOINT_NAMES, current_config)):
        print(f"  {i}. {name:15s}: {value:7.4f} rad ({np.rad2deg(value):7.2f}°)")
    
    print("\n开始测试当前配置...")
    result = test_configuration(scenario, output_dir)
    
    iteration = 0
    while True:
        print(f"\n{'='*80}")
        print(f"迭代 {iteration + 1}")
        print(f"{'='*80}")
        print("\n选项:")
        print("  t - 测试当前配置")
        print("  0-5 - 调整关节角度")
        print("  s - 保存当前配置")
        print("  q - 退出")
        
        choice = input("\n请选择: ").strip().lower()
        
        if choice == 'q':
            break
        elif choice == 's':
            config_file = output_dir / 'best_config.json'
            data = {
                'scenario_id': scenario['id'],
                'name': scenario['name'],
                'cube_pos': scenario['cube_pos'].tolist(),
                'pick_config': current_config.tolist(),
                'result': result,
            }
            with open(config_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"\n✅ 配置已保存到: {config_file}")
            print("\n保存的配置：")
            print(f"  'pick_config': np.array([{', '.join(f'{x:.4f}' for x in current_config)}]),")
        elif choice == 't':
            print("\n测试当前配置...")
            result = test_configuration(scenario, output_dir)
            iteration += 1
        elif choice.isdigit() and 0 <= int(choice) <= 5:
            joint_idx = int(choice)
            print(f"\n调整 {JOINT_NAMES[joint_idx]}:")
            print(f"  当前值: {current_config[joint_idx]:.4f} rad ({np.rad2deg(current_config[joint_idx]):.2f}°)")
            
            try:
                delta_str = input("  增量 (rad, 可以是负数): ").strip()
                delta = float(delta_str)
                current_config[joint_idx] += delta
                print(f"  新值: {current_config[joint_idx]:.4f} rad ({np.rad2deg(current_config[joint_idx]):.2f}°)")
            except ValueError:
                print("  ❌ 无效输入")
        else:
            print("❌ 无效选择")


def main():
    parser = argparse.ArgumentParser(description="单场景微调工具")
    parser.add_argument('--scenario', type=int, required=True, choices=[1,2,3,4,5],
                       help="场景ID (1-5)")
    parser.add_argument('--auto-test', action='store_true',
                       help="自动测试当前配置（不进入交互模式）")
    args = parser.parse_args()
    
    scenario = next(s for s in SCENARIOS if s['id'] == args.scenario)
    
    if args.auto_test:
        output_dir = Path(f'scenario_tuning/scenario_{scenario["id"]}')
        output_dir.mkdir(parents=True, exist_ok=True)
        test_configuration(scenario, output_dir)
    else:
        interactive_tuning(scenario)


def cli_main():
    """Dispatch tuning modes through a single public entrypoint."""
    args = sys.argv[1:]
    mode = "control"
    if "--mode" in args:
        idx = args.index("--mode")
        if idx + 1 >= len(args):
            raise SystemExit("--mode requires one of: control, live")
        mode = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    sys.argv = [sys.argv[0], *args]
    if mode == "live":
        from lerobot_sim_lab.tuning._tune_scenario_live import main as live_main

        live_main()
        return
    if mode != "control":
        raise SystemExit(f"Unsupported mode: {mode}")
    main()


if __name__ == "__main__":
    cli_main()
