#!/usr/bin/env python3
"""
比较MPlib生成的多条轨迹，检查它们的差异性
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_trajectory(file_path):
    """加载单个轨迹文件"""
    data = np.load(file_path)
    return data['trajectory']

def compare_trajectories(traj_files):
    """比较多条轨迹的差异"""
    trajectories = [load_trajectory(f) for f in traj_files]
    
    print(f"\n{'='*80}")
    print("📊 轨迹差异性分析")
    print(f"{'='*80}\n")
    
    # 基本信息
    print("基本信息:")
    for i, (traj, file) in enumerate(zip(trajectories, traj_files)):
        print(f"  Episode {i}: {len(traj)} 步 ({file.name})")
    
    # 计算两两之间的差异
    print(f"\n{'='*80}")
    print("轨迹差异对比 (MSE - Mean Squared Error):")
    print(f"{'='*80}\n")
    
    n_traj = len(trajectories)
    diff_matrix = np.zeros((n_traj, n_traj))
    
    for i in range(n_traj):
        for j in range(i+1, n_traj):
            # 对齐轨迹长度（使用较短的长度）
            min_len = min(len(trajectories[i]), len(trajectories[j]))
            traj_i = trajectories[i][:min_len]
            traj_j = trajectories[j][:min_len]
            
            # 计算均方误差
            mse = np.mean((traj_i - traj_j) ** 2)
            diff_matrix[i, j] = mse
            diff_matrix[j, i] = mse
            
            print(f"Episode {i} vs Episode {j}:")
            print(f"  MSE: {mse:.6f}")
            
            # 每个关节的差异
            joint_diffs = np.mean((traj_i - traj_j) ** 2, axis=0)
            print(f"  各关节MSE: {joint_diffs}")
            
            # 最大偏差
            max_diff = np.max(np.abs(traj_i - traj_j))
            print(f"  最大偏差: {max_diff:.6f} rad ({np.degrees(max_diff):.2f}°)")
            
            # 平均偏差
            mean_diff = np.mean(np.abs(traj_i - traj_j))
            print(f"  平均偏差: {mean_diff:.6f} rad ({np.degrees(mean_diff):.2f}°)\n")
    
    # 可视化比较
    print(f"{'='*80}")
    print("生成可视化对比图...")
    print(f"{'='*80}\n")
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('MPlib trajectory comparison - joint angles over time', fontsize=16, fontweight='bold')
    
    joint_names = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6']
    # 扩展颜色列表以支持更多轨迹
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 
              'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow',
              'navy', 'teal', 'maroon', 'lime']
    
    for joint_idx in range(6):
        ax = axes[joint_idx // 2, joint_idx % 2]
        
        for ep_idx, traj in enumerate(trajectories):
            time_steps = np.arange(len(traj))
            ax.plot(time_steps, np.degrees(traj[:, joint_idx]), 
                   label=f'Episode {ep_idx}', color=colors[ep_idx], alpha=0.7)
        
        ax.set_xlabel('time step')
        ax.set_ylabel('joint angle (°)')
        ax.set_title(f'{joint_names[joint_idx]}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = Path('pen_grab_tuning/scenario_1/visualizations/trajectory_comparison.png')
    output_file.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ 可视化图已保存: {output_file}\n")
    
    # 轨迹差异热图
    if n_traj > 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(diff_matrix, cmap='YlOrRd', aspect='auto')
        
        ax.set_xticks(np.arange(n_traj))
        ax.set_yticks(np.arange(n_traj))
        ax.set_xticklabels([f'Ep {i}' for i in range(n_traj)])
        ax.set_yticklabels([f'Ep {i}' for i in range(n_traj)])
        
        # 添加数值标注
        for i in range(n_traj):
            for j in range(n_traj):
                if i != j:
                    text = ax.text(j, i, f'{diff_matrix[i, j]:.4f}',
                                 ha="center", va="center", color="black", fontsize=10)
        
        ax.set_title('trajectory difference matrix (MSE)', fontweight='bold', fontsize=14)
        plt.colorbar(im, ax=ax, label='MSE')
        plt.tight_layout()
        
        heatmap_file = Path('pen_grab_tuning/scenario_1/visualizations/trajectory_diff_heatmap.png')
        plt.savefig(heatmap_file, dpi=150, bbox_inches='tight')
        print(f"✅ 差异热图已保存: {heatmap_file}\n")
    
    # 分析结论
    print(f"{'='*80}")
    print("结论分析:")
    print(f"{'='*80}\n")
    
    if n_traj > 1:
        # 计算所有两两对比的平均MSE
        avg_mse = np.mean([diff_matrix[i, j] for i in range(n_traj) for j in range(i+1, n_traj)])
        print(f"平均MSE: {avg_mse:.6f}")
        
        if avg_mse < 1e-6:
            print("⚠️  轨迹几乎完全相同！")
            print("   可能的原因：")
            print("   1. RRT规划失败，回退到线性插值")
            print("   2. 随机种子没有正确设置")
            print("   3. 关键帧之间的距离太近，RRT找到的路径相似")
            print("\n建议：")
            print("   1. 检查生成日志中是否有RRT失败的消息")
            print("   2. 增加 --rrt-range 参数（例如：0.2 或 0.3）")
            print("   3. 增加 --planning-time 参数（例如：5.0）")
        elif avg_mse < 0.001:
            print("⚠️  轨迹差异较小")
            print("   轨迹有一定差异，但可能不够明显")
            print("\n建议：增加 --rrt-range 来提高多样性")
        else:
            print("✅ 轨迹具有良好的多样性！")
            print(f"   平均角度偏差: {np.sqrt(avg_mse):.4f} rad ({np.degrees(np.sqrt(avg_mse)):.2f}°)")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="比较MPlib生成的轨迹")
    parser.add_argument('scenario', type=int, nargs='?', default=1,
                       help="场景ID (默认: 1)")
    parser.add_argument('--num', type=int, default=None,
                       help="比较的轨迹数量 (默认: 全部)")
    args = parser.parse_args()
    
    traj_dir = Path(f'pen_grab_tuning/scenario_{args.scenario}/trajectories')
    
    if not traj_dir.exists():
        print(f"❌ 错误: {traj_dir} 不存在")
        sys.exit(1)
    
    traj_files = sorted(traj_dir.glob('episode_*.npz'))
    
    if len(traj_files) < 2:
        print("❌ 错误: 需要至少2条轨迹才能比较")
        sys.exit(1)
    
    print(f"\n找到 {len(traj_files)} 条轨迹")
    
    # 选择要比较的轨迹数量
    if args.num:
        max_compare = min(args.num, len(traj_files))
        print(f"比较前 {max_compare} 条轨迹")
    else:
        max_compare = len(traj_files)
        print(f"比较所有 {max_compare} 条轨迹")
    
    traj_files = traj_files[:max_compare]
    
    compare_trajectories(traj_files)

