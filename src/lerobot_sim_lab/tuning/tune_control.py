#!/usr/bin/env python3
"""
场景微调控制端（与 tune_scenario_live.py 配合使用）

用法：
    终端1: python3 tune_scenario_live.py --scenario 1  (显示GUI)
    终端2: python3 tune_control.py --scenario 1         (输入命令)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

JOINT_NAMES = ['Rotation', 'Pitch', 'Elbow', 'Wrist_Pitch', 'Wrist_Roll', 'Jaw']

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
        'name': 'Left -3cm (0.03, 0.135)',
        'cube_pos': np.array([0.0300, 0.1350, 0.0170]),
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


def print_help():
    print("\n" + "="*80)
    print("微调模式 - 实时调整 pick_config")
    print("="*80)
    print("\n命令:")
    print("  0-5         - 选择要调整的关节")
    print("  +<value>    - 增加关节角度，例如: +0.05  (小步调整)")
    print("  -<value>    - 减少关节角度，例如: -0.1   (小步调整)")
    print("  ++          - 增加0.1弧度 (快捷键)")
    print("  --          - 减少0.1弧度 (快捷键)")
    print("  +           - 增加0.05弧度 (快捷键)")
    print("  -           - 减少0.05弧度 (快捷键)")
    print("  set <value> - 直接设置关节角度，例如: set -1.35")
    print("  show        - 显示当前所有关节配置")
    print("  reset       - 重置到初始配置")
    print("  save        - 保存当前配置为最佳配置")
    print("  test        - 运行完整轨迹测试（生成视频）")
    print("  h/help      - 显示此帮助")
    print("  q/quit      - 退出")
    print("\n💡 提示: 调整后VNC窗口会立即显示新姿态")
    print("="*80)
    print()


def main():
    parser = argparse.ArgumentParser(description="场景微调控制端")
    parser.add_argument('--scenario', type=int, required=True, choices=[1,2,3,4,5],
                       help="场景ID (1-5)")
    args = parser.parse_args()
    
    scenario = next(s for s in SCENARIOS if s['id'] == args.scenario)
    config_file = Path(f'scenario_tuning/scenario_{scenario["id"]}/live_config.json')
    
    if not config_file.exists():
        print(f"\n❌ 错误: {config_file} 不存在")
        print("请先在另一个终端运行:")
        print(f"  python3 tune_scenario_live.py --scenario {args.scenario}")
        return
    
    # 加载当前配置
    with open(config_file) as f:
        config_data = json.load(f)
    
    current_config = np.array(config_data['pick_config'])
    initial_config = scenario['pick_config'].copy()
    selected_joint = 0
    
    print(f"\n{'='*80}")
    print("🎯 实时微调模式")
    print(f"{'='*80}")
    print(f"场景 {scenario['id']}: {scenario['name']}")
    print(f"Cube 位置: {scenario['cube_pos']}")
    print(f"{'='*80}\n")
    print("✅ 已连接到 VNC Viewer 窗口")
    
    # 立即触发一次更新，确保viewer显示正确的初始姿态
    config_data['pick_config'] = current_config.tolist()
    config_data['mode'] = 'manual'
    config_data['updated'] = time.time()
    with open(config_file, 'w') as f:
        json.dump(config_data, f)
    print("✅ 初始配置已同步到 Viewer")
    print("\n💡 每次调整关节角度后，VNC窗口会立即更新机械臂姿态")
    print("💡 观察夹爪和cube的相对位置，调整到最佳抓取姿态\n")
    
    print_help()
    
    def show_config():
        print("\n当前配置:")
        for i, (name, value) in enumerate(zip(JOINT_NAMES, current_config)):
            marker = " <--" if i == selected_joint else ""
            print(f"  {i}. {name:15s}: {value:7.4f} rad ({np.rad2deg(value):7.2f}°){marker}")
        print()
    
    def update_viewer():
        import os
        config_data['pick_config'] = current_config.tolist()
        config_data['mode'] = 'manual'
        config_data['updated'] = time.time()
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
            f.flush()  # 刷新Python缓冲区
            os.fsync(f.fileno())  # 强制刷新到磁盘
        print("  [✓ 已发送到Viewer]", end=" ")
    
    def save_config():
        save_file = config_file.parent / 'best_config.json'
        data = {
            'scenario_id': scenario['id'],
            'name': scenario['name'],
            'cube_pos': scenario['cube_pos'].tolist(),
            'pick_config': current_config.tolist(),
        }
        with open(save_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\n✅ 配置已保存到: {save_file}")
        print("\n保存的配置：")
        print(f"  'pick_config': np.array([{', '.join(f'{x:.4f}' for x in current_config)}]),")
    
    show_config()
    
    while True:
        try:
            cmd = input(f"[关节{selected_joint}:{JOINT_NAMES[selected_joint]}] > ").strip()
            
            if not cmd:
                continue
            
            if cmd in ['quit', 'q']:
                print("退出")
                break
            elif cmd == 'help' or cmd == 'h':
                print_help()
            elif cmd == 'show':
                show_config()
            elif cmd == 'reset':
                current_config[:] = initial_config
                update_viewer()
                print("已重置到初始配置")
                show_config()
            elif cmd == 'save':
                save_config()
            elif cmd == 'test':
                print("\n🔄 运行完整轨迹测试...")
                print("(这将生成视频文件，需要等待约1分钟)")
                # 导入测试函数
                from lerobot_sim_lab.tuning.tune_scenario import test_configuration
                output_dir = config_file.parent
                result = test_configuration(scenario, current_config, output_dir, verbose=True)
                print("\n✅ 测试完成！请查看生成的视频文件")
            elif cmd.isdigit() and 0 <= int(cmd) <= 5:
                selected_joint = int(cmd)
                print(f"✅ 已选择关节 {selected_joint}: {JOINT_NAMES[selected_joint]}")
            # 快捷键
            elif cmd == '++':
                current_config[selected_joint] += 0.1
                update_viewer()
                print(f"➕ {JOINT_NAMES[selected_joint]}: {current_config[selected_joint]:.4f} rad "
                      f"({np.rad2deg(current_config[selected_joint]):.2f}°)")
            elif cmd == '--':
                current_config[selected_joint] -= 0.1
                update_viewer()
                print(f"➖ {JOINT_NAMES[selected_joint]}: {current_config[selected_joint]:.4f} rad "
                      f"({np.rad2deg(current_config[selected_joint]):.2f}°)")
            elif cmd == '+':
                current_config[selected_joint] += 0.05
                update_viewer()
                print(f"➕ {JOINT_NAMES[selected_joint]}: {current_config[selected_joint]:.4f} rad "
                      f"({np.rad2deg(current_config[selected_joint]):.2f}°)")
            elif cmd == '-':
                current_config[selected_joint] -= 0.05
                update_viewer()
                print(f"➖ {JOINT_NAMES[selected_joint]}: {current_config[selected_joint]:.4f} rad "
                      f"({np.rad2deg(current_config[selected_joint]):.2f}°)")
            elif cmd.startswith('+') or cmd.startswith('-'):
                try:
                    delta = float(cmd)
                    current_config[selected_joint] += delta
                    update_viewer()
                    print(f"🔧 {JOINT_NAMES[selected_joint]}: {current_config[selected_joint]:.4f} rad "
                          f"({np.rad2deg(current_config[selected_joint]):.2f}°)")
                except ValueError:
                    print("❌ 无效的数值")
            elif cmd.startswith('set '):
                try:
                    value = float(cmd.split()[1])
                    current_config[selected_joint] = value
                    update_viewer()
                    print(f"🎯 {JOINT_NAMES[selected_joint]}: {current_config[selected_joint]:.4f} rad "
                          f"({np.rad2deg(current_config[selected_joint]):.2f}°)")
                except (ValueError, IndexError):
                    print("❌ 无效的命令格式，使用: set <value>")
            else:
                print(f"❌ 未知命令: {cmd}")
                print("输入 'help' 查看可用命令")
        
        except KeyboardInterrupt:
            print("\n\n退出")
            break
        except Exception as e:
            print(f"❌ 错误: {e}")


if __name__ == "__main__":
    main()
