#!/usr/bin/env python3
"""
笔抓取任务多场景调参控制端（统一入口）

支持针对不同笔位置的多场景调参，每个场景独立保存 waypoints

用法：
    终端1: python -m lerobot_sim_lab.tuning.tune_pen_grab_multi --mode live --scenario 1
    终端2: python -m lerobot_sim_lab.tuning.tune_pen_grab_multi --mode control --scenario 1
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from lerobot_sim_lab.config.scenarios.pen_grab import PEN_SCENARIOS

JOINT_NAMES = ['Rotation', 'Pitch', 'Elbow', 'Wrist_Pitch', 'Wrist_Roll', 'Jaw']

# 预设的关键姿态
PRESET_POSES = {
    'home': np.array([0, -3.14, 3.14, 0.817, 0, -0.157]),  # Home姿态，夹爪张开
    'ready': np.array([0, -2.5, 2.8, 0.5, 0, 0.8]),        # 准备抓取，夹爪张开
}


def print_help():
    print("\n" + "="*80)
    print("🖊️  笔抓取任务多场景调参工具 - 命令帮助")
    print("="*80)
    print("\n【关节调整】")
    print("  0-5         - 选择要调整的关节 (0=Rotation, 1=Pitch, ..., 5=Jaw)")
    print("  +<value>    - 增加关节角度，例如: +0.05")
    print("  -<value>    - 减少关节角度，例如: -0.1")
    print("  ++          - 增加0.1弧度 (快捷键)")
    print("  --          - 减少0.1弧度 (快捷键)")
    print("  +           - 增加0.05弧度 (快捷键)")
    print("  -           - 减少0.05弧度 (快捷键)")
    print("  set <value> - 直接设置关节角度，例如: set -1.35")
    
    print("\n【关键帧管理】")
    print("  save <name> [steps]            - 保存当前姿态为关键帧，可选指定插值步数（默认50）")
    print("  save! <name> [steps]           - 强制覆盖同名关键帧，可选指定步数")
    print("  insert <name> after <ref> [steps]  - 在参考帧之后插入新帧，可选指定步数")
    print("  insert <name> before <ref> [steps] - 在参考帧之前插入新帧，可选指定步数")
    print("  waypoints                      - 显示所有已保存的关键帧")
    print("  load <name>                    - 加载姿态 (预设: home/ready, 或已保存的关键帧)")
    print("  clear                          - 清空所有关键帧")
    
    print("\n【显示和重置】")
    print("  show        - 显示当前所有关节配置")
    print("  reset       - 重置到home姿态")
    
    print("\n【其他】")
    print("  h/help      - 显示此帮助")
    print("  q/quit      - 退出")
    
    print("\n💡 关于步数 (steps):")
    print("  - 表示从前一帧到当前帧的插值步数")
    print("  - 推荐范围: 15-50")
    print("    • 15-25步: 快速动作（如松开夹爪）")
    print("    • 30-40步: 正常移动")
    print("    • 45-50步: 精细操作（如抓取、下降）")
    print("  - 如果不指定，默认使用 50 步")
    
    print("\n💡 典型工作流程:")
    print("  1. load ready       - 加载准备姿态（预设）")
    print("  2. 调整关节到接近笔的位置")
    print("  3. save 1-approach 50 - 保存第一个姿态（自动在前面添加 0-home 帧）")
    print("  4. 调整到抓取位置")
    print("  5. save 2-grasp 40    - 保存抓取姿态（40步插值）")
    print("  6. 关闭夹爪: 5, set 0.3")
    print("  7. save 3-lift 35     - 保存提起姿态（35步插值）")
    print("  8. 调整到纸盒上方")
    print("  9. save 4-place 30    - 保存放置姿态（30步插值）")
    print("  10. 打开夹爪: 5, set -0.15")
    print("  11. save 5-release 15 - 保存松开姿态（15步插值，快速）")
    print("  12. waypoints         - 查看所有关键帧（会看到自动添加的 0-home）")
    
    print("\n📹 回放和录制视频:")
    print("  在另一个终端运行:")
    print("    python -m lerobot_sim_lab.tuning.tune_pen_grab_multi --mode playback --scenario 1")
    print("    python -m lerobot_sim_lab.tuning.tune_pen_grab_multi --mode playback --scenario 1 --record")
    print("="*80)
    print()


def main():
    parser = argparse.ArgumentParser(description="笔抓取多场景调参控制端")
    parser.add_argument('--scenario', type=int, required=True, choices=[0,1,2,3,4,5],
                       help="场景ID (0-5)")
    args = parser.parse_args()
    
    scenario = next(s for s in PEN_SCENARIOS if s['id'] == args.scenario)
    
    # 每个场景独立的目录和配置文件
    scenario_dir = Path(f'pen_grab_tuning/scenario_{scenario["id"]}')
    config_file = scenario_dir / 'live_config.json'
    waypoints_file = scenario_dir / 'waypoints.json'
    
    if not config_file.exists():
        print(f"\n❌ 错误: {config_file} 不存在")
        print("请先在另一个终端运行:")
        print(f"  python -m lerobot_sim_lab.tuning.tune_pen_grab_multi --mode live --scenario {args.scenario}")
        return
    
    # 加载当前配置
    with open(config_file) as f:
        config_data = json.load(f)
    
    current_config = np.array(config_data['current_config'])
    selected_joint = 0
    
    # 加载已有的waypoints
    with open(waypoints_file) as f:
        waypoints_data = json.load(f)
    
    print(f"\n{'='*80}")
    print("🖊️  笔抓取任务多场景调参工具")
    print(f"{'='*80}")
    print(f"场景 {scenario['id']}: {scenario['name']}")
    print(f"描述: {scenario['description']}")
    print("笔位置:")
    for pen_name, (pos, quat) in scenario['pens'].items():
        print(f"  {pen_name}: {pos}")
    print("场景: assets/robots/so100/so100_6dof/scene.xml")
    print("目标: 将所有笔放入纸盒")
    print(f"配置目录: {scenario_dir}")
    print(f"{'='*80}\n")
    print("✅ 已连接到 Viewer 窗口")
    
    # 立即触发一次更新
    config_data['current_config'] = current_config.tolist()
    config_data['mode'] = 'manual'
    config_data['updated'] = time.time()
    with open(config_file, 'w') as f:
        json.dump(config_data, f)
    print("✅ 初始配置已同步到 Viewer")
    print("\n💡 每次调整后，Viewer 会立即更新机械臂姿态")
    
    # 显示已有的waypoints
    if waypoints_data['waypoints']:
        print(f"\n📍 已有 {len(waypoints_data['waypoints'])} 个关键帧:")
        for i, wp in enumerate(waypoints_data['waypoints']):
            print(f"  {i+1}. {wp['name']}")
    
    print_help()
    
    def show_config():
        print("\n当前配置:")
        for i, (name, value) in enumerate(zip(JOINT_NAMES, current_config)):
            marker = " <--" if i == selected_joint else ""
            print(f"  {i}. {name:15s}: {value:7.4f} rad ({np.rad2deg(value):7.2f}°){marker}")
        print()
    
    def update_viewer():
        config_data['current_config'] = current_config.tolist()
        config_data['mode'] = 'manual'
        config_data['updated'] = time.time()
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        print("  [✓ 已发送到Viewer]", end=" ")
    
    
    def save_waypoint(name, force_overwrite=False, steps=50):
        # 如果这是第一个用户关键帧，自动在前面添加 home 帧
        if len(waypoints_data['waypoints']) == 0:
            home_waypoint = {
                'name': '0-home',
                'config': PRESET_POSES['home'].tolist(),
                'steps': 40,  # home 帧的默认步数
                'timestamp': time.time()
            }
            waypoints_data['waypoints'].append(home_waypoint)
            print("\n✨ 自动添加 home 起始帧 (0-home)")
        
        # 检查是否已存在同名帧
        existing_idx = None
        for i, wp in enumerate(waypoints_data['waypoints']):
            if wp['name'] == name:
                existing_idx = i
                break
        
        waypoint = {
            'name': name,
            'config': current_config.tolist(),
            'steps': steps,
            'timestamp': time.time()
        }
        
        if existing_idx is not None:
            if not force_overwrite:
                confirm = input(f"⚠️  关键帧 '{name}' 已存在，是否覆盖？(y/n): ").strip().lower()
                if confirm != 'y':
                    print("❌ 取消保存")
                    return
            # 覆盖已存在的帧
            waypoints_data['waypoints'][existing_idx] = waypoint
            with open(waypoints_file, 'w', encoding='utf-8') as f:
                json.dump(waypoints_data, f, indent=2, ensure_ascii=False)
            print(f"\n✅ 关键帧已覆盖: {name} (位置: {existing_idx + 1}, 步数: {steps})")
            print(f"   当前共有 {len(waypoints_data['waypoints'])} 个关键帧")
        else:
            # 添加新的关键帧
            waypoints_data['waypoints'].append(waypoint)
            with open(waypoints_file, 'w', encoding='utf-8') as f:
                json.dump(waypoints_data, f, indent=2, ensure_ascii=False)
            print(f"\n✅ 关键帧已保存: {name} (步数: {steps})")
            print(f"   当前共有 {len(waypoints_data['waypoints'])} 个关键帧")
    
    def insert_waypoint(name, position, ref_name=None, steps=50):
        """在指定位置插入关键帧"""
        waypoint = {
            'name': name,
            'config': current_config.tolist(),
            'steps': steps,
            'timestamp': time.time()
        }
        
        if position in ['after', 'before'] and ref_name:
            # 查找参考帧的位置
            ref_idx = None
            for i, wp in enumerate(waypoints_data['waypoints']):
                if wp['name'] == ref_name:
                    ref_idx = i
                    break
            
            if ref_idx is None:
                print(f"❌ 未找到参考帧: {ref_name}")
                return
            
            insert_idx = ref_idx + 1 if position == 'after' else ref_idx
            waypoints_data['waypoints'].insert(insert_idx, waypoint)
            with open(waypoints_file, 'w', encoding='utf-8') as f:
                json.dump(waypoints_data, f, indent=2, ensure_ascii=False)
            print(f"\n✅ 关键帧已插入: {name} ({position} '{ref_name}', 位置: {insert_idx + 1}, 步数: {steps})")
            print(f"   当前共有 {len(waypoints_data['waypoints'])} 个关键帧")
        else:
            print("❌ 无效的插入命令")
            print("   用法: insert <name> after <ref_name> [steps]")
            print("        insert <name> before <ref_name> [steps]")
    
    def show_waypoints():
        if not waypoints_data['waypoints']:
            print("\n📍 还没有保存任何关键帧")
            return
        print(f"\n📍 已保存的关键帧 ({len(waypoints_data['waypoints'])}个):")
        for i, wp in enumerate(waypoints_data['waypoints'], 1):
            config = np.array(wp['config'])
            steps = wp.get('steps', 50)  # 兼容旧格式，默认50步
            print(f"\n  {i}. {wp['name']} (步数: {steps})")
            for j, (jname, val) in enumerate(zip(JOINT_NAMES, config)):
                print(f"     {jname:15s}: {val:7.4f} rad ({np.rad2deg(val):7.2f}°)")
        print()
    
    def clear_waypoints():
        confirm = input("⚠️  确定要清空所有关键帧吗？(yes/no): ").strip().lower()
        if confirm == 'yes':
            waypoints_data['waypoints'] = []
            with open(waypoints_file, 'w', encoding='utf-8') as f:
                json.dump(waypoints_data, f, indent=2, ensure_ascii=False)
            print("✅ 所有关键帧已清空")
        else:
            print("❌ 取消操作")
    
    def load_preset(name):
        # 先检查是否是预设姿态
        if name in PRESET_POSES:
            current_config[:] = PRESET_POSES[name]
            update_viewer()
            print(f"\n✅ 已加载预设姿态: {name}")
            show_config()
        else:
            # 再检查是否是已保存的关键帧
            found = False
            for wp in waypoints_data['waypoints']:
                if wp['name'] == name:
                    current_config[:] = np.array(wp['config'])
                    update_viewer()
                    print(f"\n✅ 已加载关键帧: {name}")
                    show_config()
                    found = True
                    break
            
            if not found:
                print(f"❌ 未找到姿态: {name}")
                print(f"   可用预设: {', '.join(PRESET_POSES.keys())}")
                if waypoints_data['waypoints']:
                    saved_names = [wp['name'] for wp in waypoints_data['waypoints']]
                    print(f"   已保存关键帧: {', '.join(saved_names)}")
    
    show_config()
    
    while True:
        try:
            cmd = input(f"[场景{scenario['id']}][关节{selected_joint}:{JOINT_NAMES[selected_joint]}] > ").strip()
            
            if not cmd:
                continue
            
            if cmd in ['quit', 'q']:
                print("退出")
                break
            elif cmd in ['help', 'h']:
                print_help()
            elif cmd == 'show':
                show_config()
            elif cmd == 'reset':
                current_config[:] = PRESET_POSES['home']
                update_viewer()
                print("已重置到home姿态")
                show_config()
            elif cmd == 'waypoints':
                show_waypoints()
            elif cmd == 'clear':
                clear_waypoints()
            elif cmd.startswith('save! '):
                # 强制覆盖
                try:
                    parts = cmd.split()[1:]
                    if len(parts) >= 1:
                        name = parts[0]
                        steps = int(parts[1]) if len(parts) >= 2 else 50
                        if steps < 10 or steps > 200:
                            print(f"⚠️  步数 {steps} 超出推荐范围 (10-200)，使用默认值 50")
                            steps = 50
                        save_waypoint(name, force_overwrite=True, steps=steps)
                    else:
                        print("❌ 无效的命令格式")
                        print("   用法: save! <name> [steps]")
                except ValueError:
                    print("❌ 步数必须是整数")
            elif cmd.startswith('save '):
                try:
                    parts = cmd.split()[1:]
                    if len(parts) >= 1:
                        name = parts[0]
                        steps = int(parts[1]) if len(parts) >= 2 else 50
                        if steps < 10 or steps > 200:
                            print(f"⚠️  步数 {steps} 超出推荐范围 (10-200)，使用默认值 50")
                            steps = 50
                        save_waypoint(name, steps=steps)
                    else:
                        print("❌ 无效的命令格式")
                except ValueError:
                    print("❌ 步数必须是整数")
            elif cmd.startswith('insert '):
                try:
                    parts = cmd.split()
                    if len(parts) >= 4 and parts[2] in ['after', 'before']:
                        name = parts[1]
                        position = parts[2]
                        ref_name = parts[3]
                        steps = int(parts[4]) if len(parts) >= 5 else 50
                        if steps < 10 or steps > 200:
                            print(f"⚠️  步数 {steps} 超出推荐范围 (10-200)，使用默认值 50")
                            steps = 50
                        insert_waypoint(name, position, ref_name, steps=steps)
                    else:
                        print("❌ 无效的 insert 命令格式")
                except ValueError:
                    print("❌ 步数必须是整数")
            elif cmd.startswith('load '):
                name = cmd.split(maxsplit=1)[1]
                load_preset(name)
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
    args = sys.argv[1:]
    mode = "control"
    if "--mode" in args:
        idx = args.index("--mode")
        if idx + 1 >= len(args):
            raise SystemExit("--mode requires one of: control, live, playback")
        mode = args[idx + 1]
        args = args[:idx] + args[idx + 2:]
        sys.argv = [sys.argv[0], *args]

    if mode == "live":
        from lerobot_sim_lab.tuning._tune_pen_grab_multi_live import main as live_main

        live_main()
    elif mode == "playback":
        from lerobot_sim_lab.tuning._tune_pen_grab_multi_playback import main as playback_main

        playback_main()
    elif mode == "control":
        main()
    else:
        raise SystemExit(f"Unsupported mode: {mode}")
