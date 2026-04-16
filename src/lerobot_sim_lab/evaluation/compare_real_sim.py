#!/usr/bin/env python3
"""
对比真实场景和仿真场景的机械臂初始姿态

用法：
    python3 compare_real_sim_init.py <真实数据集路径>
    
示例：
    python3 compare_real_sim_init.py create
    python3 compare_real_sim_init.py create --frame 10
    
注意：
    - 数据集路径应该是包含 meta_data/ 和 data/ 的目录
    - 不要加 /data 后缀
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

JOINT_NAMES = ['Rotation', 'Pitch', 'Elbow', 'Wrist_Pitch', 'Wrist_Roll', 'Jaw']

SIM_HOME = np.array([0, -3.14, 3.14, 0.817, 0, -0.157])


def print_joint_angles(angles, title, joint_names=None):
    """打印关节角度"""
    if joint_names is None:
        joint_names = JOINT_NAMES
    
    print(f"\n{title}")
    for i, (name, val) in enumerate(zip(joint_names, angles)):
        print(f'  {i}. {name:15s}: {val:8.4f} rad  ({np.rad2deg(val):7.2f}°)')


def load_dataset_frame(dataset_path, frame_idx=0):
    """直接从 parquet 文件加载指定帧的数据"""
    print('=' * 80)
    print('📂 加载真实场景数据集...')
    print('=' * 80)
    
    dataset_path_obj = Path(dataset_path)
    data_dir = dataset_path_obj / 'data'
    
    if not data_dir.exists():
        print(f'❌ 数据目录不存在: {data_dir}')
        return None
    
    parquet_files = sorted(data_dir.glob('chunk-*/file-*.parquet'))
    
    if not parquet_files:
        print('❌ 未找到 parquet 文件')
        return None
    
    print(f'✅ 找到 {len(parquet_files)} 个 parquet 文件')
    print(f'✅ 读取: {parquet_files[0]}')
    
    df = pd.read_parquet(parquet_files[0])
    
    print(f'✅ 总帧数: {len(df)}')
    
    if frame_idx >= len(df):
        print(f'❌ 帧索引超出范围: {frame_idx} >= {len(df)}')
        return None
    
    row = df.iloc[frame_idx]
    
    return row


def main():
    parser = argparse.ArgumentParser(description="对比真实场景和仿真场景的机械臂初始姿态")
    parser.add_argument('dataset_path', type=str, nargs='?', default='create',
                       help="真实数据集路径（默认: create，注意：不要加 /data 后缀）")
    parser.add_argument('--frame', type=int, default=0,
                       help="查看第N帧（默认: 0，即第一帧）")
    args = parser.parse_args()
    
    frame = load_dataset_frame(args.dataset_path, args.frame)
    
    if frame is None:
        return
    
    print('\n' + '=' * 80)
    print(f'🎬 第 {args.frame} 帧数据')
    print('=' * 80)
    
    print('\n数据概览:')
    for key, value in frame.items():
        if isinstance(value, (np.ndarray, list)):
            value_array = np.array(value)
            if value_array.size <= 10:
                print(f'  {key}: {value_array}')
            else:
                print(f'  {key}: shape={value_array.shape}')
        else:
            print(f'  {key}: {value}')
    
    print('\n' + '=' * 80)
    print(f'🤖 机械臂状态 (第 {args.frame} 帧)')
    print('=' * 80)
    
    state_np = np.array(frame['observation.state'])
    action_np = np.array(frame['action'])
    
    # 数据是以度为单位存储的，需要转换为弧度
    state_rad = np.deg2rad(state_np)
    action_rad = np.deg2rad(action_np)
    
    print('\nobservation.state:')
    print(f'  原始值 (度): {state_np}')
    print(f'  转换为弧度: {state_rad}')
    print(f'  长度: {len(state_np)}')
    
    if len(state_np) >= 6:
        print_joint_angles(state_rad[:6], '  前6个关节 (6-DOF):')
    
    print('\naction:')
    print(f'  原始值 (度): {action_np}')
    print(f'  转换为弧度: {action_rad}')
    print(f'  长度: {len(action_np)}')
    
    if len(action_np) >= 6:
        print_joint_angles(action_rad[:6], '  前6个关节 (6-DOF):')
    
    print('\n' + '=' * 80)
    print('📝 对比：仿真场景的 home 姿态')
    print('=' * 80)
    
    print_joint_angles(SIM_HOME, '仿真 home 姿态:')
    
    if len(state_np) >= 6:
        print('\n' + '=' * 80)
        print('⚖️  差异分析 (真实 - 仿真)')
        print('=' * 80)
        
        real_angles = state_rad[:6]
        
        print('\n差异值:')
        max_diff_deg = 0
        for i, (name, real_val, sim_val) in enumerate(zip(JOINT_NAMES, real_angles, SIM_HOME)):
            diff = real_val - sim_val
            diff_deg = np.rad2deg(diff)
            max_diff_deg = max(max_diff_deg, abs(diff_deg))
            
            if abs(diff_deg) > 5:
                marker = ' ❗ 差异较大'
            elif abs(diff_deg) > 1:
                marker = ' ⚠️  需注意'
            else:
                marker = ' ✓ 接近'
            print(f'  {i}. {name:15s}: {diff:8.4f} rad  ({diff_deg:7.2f}°){marker}')
        
        print('\n' + '=' * 80)
        print('💡 建议')
        print('=' * 80)
        
        if max_diff_deg < 1:
            print('✅ 真实场景和仿真场景的初始姿态非常接近！')
        elif max_diff_deg < 5:
            print('⚠️  真实场景和仿真场景的初始姿态有轻微差异。')
            print('   建议：')
            print('   1. 可以接受这个差异，继续使用当前设置')
            print('   2. 或者微调仿真场景的 home 姿态')
        else:
            print('❗ 真实场景和仿真场景的初始姿态差异较大！')
            print('   强烈建议：')
            print('   1. 更新 waypoints.json 中的 0-home 帧为真实场景的初始姿态')
            print('   2. 或者调整真实机械臂的初始姿态，使其与仿真 home 姿态一致')
            print('\n   真实场景的初始姿态（可直接复制到 waypoints.json）:')
            print(f'   config (弧度): {real_angles.tolist()}')
            print(f'   config (度): {state_np[:6].tolist()}')
    
    print('\n' + '=' * 80)


if __name__ == '__main__':
    main()
