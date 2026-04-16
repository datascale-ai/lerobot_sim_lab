#!/usr/bin/env python3
"""
可视化MPlib生成的多样化轨迹

对比：
1. 线性插值（当前方案）vs MPlib规划（改进方案）
2. 显示不同episode之间的路径差异
3. 计算多样性指标
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def linear_interpolate_trajectory(waypoints, steps_per_segment=50):
    """线性插值生成轨迹（对比基线）"""
    trajectory = []
    for i in range(len(waypoints) - 1):
        start = np.array(waypoints[i]['config'])
        end = np.array(waypoints[i+1]['config'])
        steps = waypoints[i+1].get('steps', steps_per_segment)
        
        for step in range(steps):
            alpha = step / steps
            config = start + alpha * (end - start)
            trajectory.append(config)
    
    trajectory.append(np.array(waypoints[-1]['config']))
    return np.array(trajectory)


def compute_diversity_metrics(episodes):
    """计算轨迹多样性指标"""
    num_episodes = len(episodes)
    
    # 1. 轨迹长度多样性
    lengths = [len(traj) for traj in episodes]
    length_diversity = np.std(lengths) / (np.mean(lengths) + 1e-8)
    
    # 2. 路径偏差（每个时刻的标准差平均值）
    # 首先对齐所有轨迹长度（插值到相同长度）
    max_len = max(lengths)
    aligned_trajs = []
    for traj in episodes:
        if len(traj) < max_len:
            # 线性插值到max_len
            indices = np.linspace(0, len(traj)-1, max_len)
            aligned = np.array([np.interp(indices, np.arange(len(traj)), traj[:, j]) 
                               for j in range(traj.shape[1])]).T
        else:
            aligned = traj
        aligned_trajs.append(aligned)
    
    aligned_trajs = np.array(aligned_trajs)  # [num_episodes, max_len, 6]
    
    # 计算每个时刻的标准差
    path_std = np.std(aligned_trajs, axis=0)  # [max_len, 6]
    mean_path_deviation = np.mean(path_std)
    
    # 3. 成对轨迹距离（DTW或欧氏距离）
    pairwise_distances = []
    for i in range(num_episodes):
        for j in range(i+1, num_episodes):
            # 简化：使用对齐后的欧氏距离
            dist = np.linalg.norm(aligned_trajs[i] - aligned_trajs[j])
            pairwise_distances.append(dist)
    
    mean_pairwise_dist = np.mean(pairwise_distances) if pairwise_distances else 0
    
    return {
        'num_episodes': num_episodes,
        'length_diversity': length_diversity,
        'mean_path_deviation': mean_path_deviation,
        'mean_pairwise_distance': mean_pairwise_dist,
        'length_range': (min(lengths), max(lengths)),
        'length_mean_std': (np.mean(lengths), np.std(lengths))
    }


def visualize_trajectories(episodes, waypoints, output_file='trajectory_comparison.png'):
    """可视化多条轨迹的对比"""
    num_episodes = len(episodes)
    num_joints = episodes[0].shape[1]
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'MPlib生成的多样化轨迹对比 ({num_episodes} episodes)', fontsize=16)
    
    joint_names = ['Rotation', 'Pitch', 'Elbow', 'Wrist_Pitch', 'Wrist_Roll', 'Jaw']
    
    for joint_idx in range(num_joints):
        ax = axes[joint_idx // 2, joint_idx % 2]
        
        # 绘制每个episode的轨迹
        for ep_idx, traj in enumerate(episodes):
            time_steps = np.arange(len(traj))
            alpha = 0.6 if num_episodes > 5 else 0.8
            ax.plot(time_steps, traj[:, joint_idx], 
                   alpha=alpha, linewidth=1.5,
                   label=f'Episode {ep_idx+1}' if num_episodes <= 5 else None)
        
        # 标记关键帧位置
        waypoint_configs = [np.array(wp['config']) for wp in waypoints]
        waypoint_positions = [0]  # 第一个关键帧在t=0
        cumulative_steps = 0
        for i in range(1, len(waypoints)):
            cumulative_steps += waypoints[i].get('steps', 50)
            waypoint_positions.append(cumulative_steps)
        
        for pos, wp in zip(waypoint_positions, waypoints):
            ax.axvline(pos, color='red', linestyle='--', alpha=0.3, linewidth=1)
            ax.plot(pos, waypoint_configs[waypoint_positions.index(pos)][joint_idx], 
                   'ro', markersize=8, markeredgecolor='darkred', markeredgewidth=1.5)
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Joint Angle (rad)')
        ax.set_title(f'{joint_names[joint_idx]}')
        ax.grid(True, alpha=0.3)
        
        if num_episodes <= 5 and joint_idx == 0:
            ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ 轨迹对比图已保存: {output_file}")
    plt.close()


def visualize_linear_vs_mplib(waypoints, mplib_episodes, output_file='linear_vs_mplib.png'):
    """对比线性插值 vs MPlib规划"""
    # 生成一条线性插值轨迹
    linear_traj = linear_interpolate_trajectory(waypoints)
    
    # 选择第一条MPlib轨迹作为代表
    mplib_traj = mplib_episodes[0]
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('线性插值 vs MPlib运动规划对比', fontsize=16)
    
    joint_names = ['Rotation', 'Pitch', 'Elbow', 'Wrist_Pitch', 'Wrist_Roll', 'Jaw']
    
    for joint_idx in range(6):
        ax = axes[joint_idx // 2, joint_idx % 2]
        
        # 线性插值
        time_linear = np.arange(len(linear_traj))
        ax.plot(time_linear, linear_traj[:, joint_idx], 
               'b-', linewidth=2, label='线性插值', alpha=0.7)
        
        # MPlib规划
        time_mplib = np.arange(len(mplib_traj))
        ax.plot(time_mplib, mplib_traj[:, joint_idx], 
               'g-', linewidth=2, label='MPlib规划', alpha=0.7)
        
        # 标记关键帧
        waypoint_configs = [np.array(wp['config']) for wp in waypoints]
        waypoint_positions_linear = [0]
        cumulative = 0
        for i in range(1, len(waypoints)):
            cumulative += waypoints[i].get('steps', 50)
            waypoint_positions_linear.append(cumulative)
        
        for pos, wp_config in zip(waypoint_positions_linear, waypoint_configs):
            ax.plot(pos, wp_config[joint_idx], 
                   'ro', markersize=10, markeredgecolor='darkred', 
                   markeredgewidth=2, label='关键帧' if joint_idx == 0 and pos == waypoint_positions_linear[0] else None)
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Joint Angle (rad)')
        ax.set_title(f'{joint_names[joint_idx]}')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ 对比图已保存: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="可视化轨迹多样性")
    parser.add_argument('--scenario', type=int, required=True, choices=[1,2,3,4,5])
    args = parser.parse_args()
    
    scenario_dir = Path(f'pen_grab_tuning/scenario_{args.scenario}')
    waypoints_file = scenario_dir / 'waypoints.json'
    mplib_dir = scenario_dir / 'mplib_trajectories'
    
    if not mplib_dir.exists():
        print(f"❌ 错误: {mplib_dir} 不存在")
        print(f"   请先运行: python3 mplib_trajectory_generator.py --scenario {args.scenario}")
        return
    
    # 加载关键帧
    with open(waypoints_file) as f:
        waypoints_data = json.load(f)
    waypoints = waypoints_data['waypoints']
    
    # 加载MPlib生成的轨迹
    mplib_files = sorted(mplib_dir.glob('episode_*.npz'))
    mplib_episodes = []
    for f in mplib_files:
        data = np.load(f)
        mplib_episodes.append(data['trajectory'])
    
    print(f"\n{'='*80}")
    print("📊 轨迹多样性分析")
    print(f"{'='*80}")
    print(f"场景: {args.scenario}")
    print(f"关键帧数量: {len(waypoints)}")
    print(f"MPlib生成的episode数量: {len(mplib_episodes)}")
    print(f"{'='*80}\n")
    
    # 计算多样性指标
    metrics = compute_diversity_metrics(mplib_episodes)
    
    print("多样性指标:")
    print(f"  轨迹长度范围: {metrics['length_range'][0]} - {metrics['length_range'][1]} 步")
    print(f"  轨迹长度均值±标准差: {metrics['length_mean_std'][0]:.1f} ± {metrics['length_mean_std'][1]:.1f}")
    print(f"  长度多样性系数: {metrics['length_diversity']:.4f}")
    print(f"  平均路径偏差: {metrics['mean_path_deviation']:.4f} rad")
    print(f"  平均成对距离: {metrics['mean_pairwise_distance']:.2f}")
    print()
    
    # 生成可视化
    output_dir = scenario_dir / 'visualizations'
    output_dir.mkdir(exist_ok=True)
    
    print("生成可视化图表...")
    visualize_trajectories(
        mplib_episodes, 
        waypoints,
        output_file=output_dir / 'mplib_diversity.png'
    )
    
    visualize_linear_vs_mplib(
        waypoints,
        mplib_episodes,
        output_file=output_dir / 'linear_vs_mplib.png'
    )
    
    print(f"\n✅ 所有可视化已保存到: {output_dir}")


if __name__ == "__main__":
    main()


