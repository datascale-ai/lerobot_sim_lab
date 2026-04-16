#!/usr/bin/env python3
"""
回放MPlib生成的多样化轨迹

支持：
- 单个episode回放
- 连续回放多个episodes对比
- 录制视频
- 调整回放速度
"""

import argparse
import time
from pathlib import Path

import cv2
import imageio
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

BOX_BOUNDS = {
    "x": (-0.06, 0.06),
    "y": (-0.08, 0.08),
    "z": (0.0, 0.08),
}
BOX_MARGIN = 0.008


def set_pens_positions(model, data, pens_config):
    """设置场景中所有笔的位置"""
    data.qpos[BOX_QPOS_START:BOX_QPOS_START+3] = BOX_POSITION
    data.qpos[BOX_QPOS_START+3:BOX_QPOS_START+7] = BOX_QUATERNION
    
    for pen_name, (pos, quat) in pens_config.items():
        if pen_name in PEN_QPOS_MAP:
            qpos_start = PEN_QPOS_MAP[pen_name]
            data.qpos[qpos_start:qpos_start+3] = pos
            data.qpos[qpos_start+3:qpos_start+7] = quat
    
    mujoco.mj_forward(model, data)


def load_trajectory(episode_file: Path):
    """加载单个MPlib生成的轨迹"""
    data = np.load(episode_file)
    trajectory = data['trajectory']  # [T, 6]
    return trajectory


def _get_body_id(model, name: str):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return body_id if body_id >= 0 else None


def _in_box(local_pos):
    x, y, z = local_pos
    return (BOX_BOUNDS["x"][0] - BOX_MARGIN <= x <= BOX_BOUNDS["x"][1] + BOX_MARGIN and
            BOX_BOUNDS["y"][0] - BOX_MARGIN <= y <= BOX_BOUNDS["y"][1] + BOX_MARGIN and
            BOX_BOUNDS["z"][0] - BOX_MARGIN <= z <= BOX_BOUNDS["z"][1] + BOX_MARGIN)


def _get_pen_geom_pos(model, data, pen_body_id: int):
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) != pen_body_id:
            continue
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            return data.geom_xpos[geom_id].copy()
    return data.xpos[pen_body_id].copy()


def check_pens_in_box(model, data):
    box_id = _get_body_id(model, "paper_box")
    pen_names = ["pen1", "pen_2", "pen_3", "pen_4"]
    if box_id is None:
        return False, []
    box_pos = data.xpos[box_id].copy()
    box_mat = data.xmat[box_id].reshape(3, 3)
    inside = []
    for name in pen_names:
        pen_id = _get_body_id(model, name)
        if pen_id is None:
            inside.append(False)
            continue
        pen_pos = _get_pen_geom_pos(model, data, pen_id)
        local = box_mat.T @ (pen_pos - box_pos)
        inside.append(_in_box(local))
    return all(inside), inside


def playback_single_episode(model, data, trajectory, scenario, 
                           playback_speed=1.0, record=False, output_file=None):
    """回放单个episode"""
    print(f"\n{'='*80}")
    print(f"回放轨迹: {len(trajectory)} 步")
    print(f"回放速度: {playback_speed}x")
    if record:
        print(f"录制视频: {output_file}")
    print(f"{'='*80}\n")
    
    frames = [] if record else None
    renderer = mujoco.Renderer(model, height=480, width=640) if record else None
    
    # 计算每个控制步的物理子步数（30Hz控制频率）
    n_substeps = int((1/30) / model.opt.timestep)
    print(f"物理仿真设置: 控制频率 30Hz, 物理步/控制步: {n_substeps}\n")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 设置相机视角（类似camera_front_new）
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'Base')
        if base_body_id >= 0:
            viewer.cam.lookat[:] = data.xpos[base_body_id]
        else:
            viewer.cam.lookat[:] = [0.28, 0.08, 0.75]
        viewer.cam.distance = 0.8
        viewer.cam.azimuth = -90
        viewer.cam.elevation = -45
        
        # 如果录制，同步renderer的相机设置
        if record:
            renderer.update_scene(data, camera='camera_front_new')
        
        # 初始化到home姿态
        home_keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
        if home_keyframe_id >= 0:
            mujoco.mj_resetDataKeyframe(model, data, home_keyframe_id)
        else:
            mujoco.mj_resetData(model, data)
        
        # 设置场景中的笔和物体位置
        set_pens_positions(model, data, scenario['pens'])
        
        # 设置机械臂到轨迹起始位置
        data.ctrl[:6] = trajectory[0]
        data.qpos[:6] = trajectory[0]
        mujoco.mj_forward(model, data)
        
        print("开始回放...")
        start_time = time.time()
        
        for step_idx, qpos in enumerate(trajectory):
            if not viewer.is_running():
                print("\n用户关闭了viewer，停止回放")
                break
            
            # 设置机械臂关节目标位置（通过控制器）
            data.ctrl[:6] = qpos
            
            # 运行多个物理步骤（让控制器响应并进行物理交互）
            for substep in range(n_substeps):
                mujoco.mj_step(model, data)
                
                # 只在最后一个子步更新viewer，减少渲染负担
                if substep == n_substeps - 1:
                    viewer.sync()
            
            # 录制帧（每个控制步录制一帧，30Hz）
            if record:
                renderer.update_scene(data, camera='camera_front_new')
                pixels = renderer.render()
                frames.append(pixels)
            
            # 进度显示
            if (step_idx + 1) % 50 == 0:
                progress = (step_idx + 1) / len(trajectory) * 100
                print(f"  进度: {progress:.1f}% ({step_idx + 1}/{len(trajectory)})")
        
        elapsed = time.time() - start_time
        print("\n✅ 回放完成！")
        print(f"   总时间: {elapsed:.2f}秒")
        print(f"   实际速度: {len(trajectory) * 0.01 / elapsed:.2f}x")
        success, inside = check_pens_in_box(model, data)
        print(f"   放置结果: {'成功' if success else '失败'} (pen1/2/3/4: {inside})")
    
    # 保存视频
    if record and frames:
        print(f"\n正在保存视频到: {output_file}")
        imageio.mimsave(output_file, frames, fps=50)
        print("✅ 视频已保存！")
        print(f"   帧数: {len(frames)}")
        print(f"   分辨率: {frames[0].shape[:2]}")
    
    # 清理
    if renderer:
        renderer.close()
    return success


def playback_quad_view(model, data, trajectories, scenario, episode_indices, 
                       output_file=None, playback_speed=1.0):
    """
    四分屏回放并录制为视频
    同时显示4个episode，拼接成2x2网格
    """
    print(f"\n{'='*80}")
    print("🎬 四分屏回放录制")
    print(f"Episodes: {episode_indices}")
    print(f"输出: {output_file}")
    print(f"{'='*80}\n")
    
    # 确保有4个轨迹（不足的用第一个补充）
    while len(trajectories) < 4:
        trajectories.append(trajectories[0])
        episode_indices.append(episode_indices[0])
    trajectories = trajectories[:4]
    episode_indices = episode_indices[:4]
    
    # 不对齐轨迹长度，保留原始差异以突出对比性
    max_len = max(len(traj) for traj in trajectories)
    traj_lengths = [len(traj) for traj in trajectories]
    
    print("轨迹长度差异:")
    for i, (traj, ep_idx) in enumerate(zip(trajectories, episode_indices)):
        print(f"  Episode {ep_idx}: {len(traj)} 步 ({len(traj)*0.01:.2f}秒)")
    print(f"最长轨迹: {max_len} 步 ({max_len*0.01:.2f}秒)\n")
    
    # 不做插值对齐，保持原始长度
    aligned_trajs = trajectories
    
    # 创建4个渲染器（每个episode一个）
    renderers = [mujoco.Renderer(model, height=480, width=640) for _ in range(4)]
    
    # 创建4个独立的data副本
    data_copies = [mujoco.MjData(model) for _ in range(4)]
    
    # 初始化所有场景
    for d in data_copies:
        home_keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
        if home_keyframe_id >= 0:
            mujoco.mj_resetDataKeyframe(model, d, home_keyframe_id)
        else:
            mujoco.mj_resetData(model, d)
        set_pens_positions(model, d, scenario['pens'])
    
    frames = []
    n_substeps = int((1/30) / model.opt.timestep)
    
    print("开始录制四分屏视频...\n")
    
    # 记录每个episode是否已完成
    finished = [False] * 4
    
    for step_idx in range(max_len):
        # 更新每个episode的状态
        for ep_idx, (traj, d, traj_len) in enumerate(zip(aligned_trajs, data_copies, traj_lengths)):
            # 如果轨迹还没播放完，正常更新
            if step_idx < traj_len:
                d.ctrl[:6] = traj[step_idx]
            else:
                # 轨迹已播放完，保持在最后一帧
                if not finished[ep_idx]:
                    finished[ep_idx] = True
                    print(f"  ✅ Episode {episode_indices[ep_idx]} 完成 (第{step_idx}步)")
                # 继续保持最后一帧的控制信号
                d.ctrl[:6] = traj[-1]
            
            # 运行物理仿真
            for _ in range(n_substeps):
                mujoco.mj_step(model, d)
        
        # 渲染4个视角
        sub_frames = []
        for ep_idx, (renderer, d, traj_len) in enumerate(zip(renderers, data_copies, traj_lengths)):
            renderer.update_scene(d, camera='camera_front_new')
            frame = renderer.render()
            
            # 在帧上添加文字标签
            frame = frame.copy()
            
            # Episode编号（左上角）
            text = f"Episode {episode_indices[ep_idx]}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, (255, 255, 0), 2, cv2.LINE_AA)
            
            # 时间步信息（左下角）
            current_step = min(step_idx + 1, traj_len)
            time_text = f"{current_step}/{traj_len} ({current_step*0.01:.2f}s)"
            cv2.putText(frame, time_text, (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
            # 如果已完成，添加FINISHED标签（中央，绿色）
            if step_idx >= traj_len:
                finished_text = "FINISHED"
                text_size = cv2.getTextSize(finished_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                text_x = (frame.shape[1] - text_size[0]) // 2
                text_y = frame.shape[0] // 2
                
                # 添加半透明背景
                overlay = frame.copy()
                cv2.rectangle(overlay, 
                            (text_x - 10, text_y - text_size[1] - 10),
                            (text_x + text_size[0] + 10, text_y + 10),
                            (0, 150, 0), -1)
                frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
                
                # 添加文字
                cv2.putText(frame, finished_text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
            
            sub_frames.append(frame)
        
        # 拼接成2x2网格
        top_row = np.hstack([sub_frames[0], sub_frames[1]])
        bottom_row = np.hstack([sub_frames[2], sub_frames[3]])
        quad_frame = np.vstack([top_row, bottom_row])
        frames.append(quad_frame)
        
        # 进度显示
        if (step_idx + 1) % 50 == 0:
            progress = (step_idx + 1) / max_len * 100
            print(f"  录制进度: {progress:.1f}% ({step_idx + 1}/{max_len})")
    
    # 清理渲染器
    for renderer in renderers:
        renderer.close()
    
    results = []
    for d in data_copies:
        success, inside = check_pens_in_box(model, d)
        results.append((success, inside))
    print("\n放置结果:")
    for ep_idx, (success, inside) in zip(episode_indices, results):
        print(f"  Episode {ep_idx}: {'成功' if success else '失败'} (pen1/2/3/4: {inside})")
    
    # 保存视频
    if output_file and frames:
        print(f"\n正在保存四分屏视频到: {output_file}")
        imageio.mimsave(output_file, frames, fps=30)
        print("✅ 视频已保存！")
        print(f"   帧数: {len(frames)}")
        print(f"   分辨率: {frames[0].shape[:2]} (2x2 网格)")
        print(f"   文件大小: {Path(output_file).stat().st_size / 1024 / 1024:.1f} MB")
    
    print(f"\n{'='*80}\n")


def playback_comparison(model, data, trajectories, scenario, episode_indices):
    """对比回放多个episodes（并排显示）"""
    print(f"\n{'='*80}")
    print(f"对比回放 {len(trajectories)} 条轨迹")
    print(f"Episode索引: {episode_indices}")
    print(f"{'='*80}\n")
    
    # 找到最长的轨迹
    max_len = max(len(traj) for traj in trajectories)
    
    # 对齐所有轨迹长度（通过插值）
    aligned_trajs = []
    for traj in trajectories:
        if len(traj) < max_len:
            indices = np.linspace(0, len(traj)-1, max_len)
            aligned = np.array([np.interp(indices, np.arange(len(traj)), traj[:, j]) 
                               for j in range(traj.shape[1])]).T
        else:
            aligned = traj
        aligned_trajs.append(aligned)
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 设置相机
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = [0.28, 0.08, 0.75]
        viewer.cam.distance = 1.2
        viewer.cam.azimuth = -90
        viewer.cam.elevation = -45
        
        set_pens_positions(model, data, scenario['pens'])
        
        print("按空格键开始/暂停回放...")
        print("按 'r' 重新开始")
        print("按 'q' 或关闭窗口退出\n")
        
        current_episode = 0
        step_idx = 0
        paused = True
        
        while viewer.is_running():
            if not paused and step_idx < max_len:
                # 显示当前episode的轨迹
                qpos = aligned_trajs[current_episode][step_idx]
                data.qpos[:6] = qpos
                data.ctrl[:6] = qpos
                set_pens_positions(model, data, scenario['pens'])
                mujoco.mj_forward(model, data)
                
                step_idx += 1
                
                if step_idx >= max_len:
                    # 切换到下一个episode
                    current_episode = (current_episode + 1) % len(trajectories)
                    step_idx = 0
                    print(f"\n切换到 Episode {episode_indices[current_episode]}")
                    time.sleep(1.0)
                
                if step_idx % 50 == 0:
                    print(f"  Episode {episode_indices[current_episode]}: {step_idx}/{max_len}")
            
            viewer.sync()
            time.sleep(0.01)


def main():
    parser = argparse.ArgumentParser(description="回放MPlib生成的轨迹")
    parser.add_argument('--scenario', type=int, required=True, choices=[0,1,2,3,4,5],
                       help="场景ID")
    parser.add_argument('--episode', type=int, default=None,
                       help="回放指定episode（默认：回放所有）")
    parser.add_argument('--episodes', type=int, nargs='+', default=None,
                       help="回放多个episodes对比，例如: --episodes 0 1 2")
    parser.add_argument('--speed', type=float, default=1.0,
                       help="回放速度倍数（默认：1.0）")
    parser.add_argument('--record', action='store_true',
                       help="录制视频")
    parser.add_argument('--compare', action='store_true',
                       help="对比模式（循环显示多个episodes）")
    parser.add_argument('--quad-view', action='store_true',
                       help="四分屏录制模式（同时录制4个episodes为2x2网格视频）")
    parser.add_argument('--save-success-seeds', action='store_true',
                       help="回放所有轨迹后保存成功episode索引到seed.txt")
    args = parser.parse_args()
    
    # 加载场景
    scenario = next(s for s in PEN_SCENARIOS if s['id'] == args.scenario)
    scenario_dir = Path(f'pen_grab_tuning/scenario_{args.scenario}')
    mplib_dir = scenario_dir / 'trajectories'
    
    if not mplib_dir.exists():
        print(f"❌ 错误: {mplib_dir} 不存在")
        print(f"   请先运行: python3 trajectory_generator.py --scenario {args.scenario}")
        return
    
    # 加载轨迹文件
    episode_files = sorted(mplib_dir.glob('episode_*.npz'))
    if not episode_files:
        print("❌ 错误: 没有找到轨迹文件")
        return
    
    print(f"\n{'='*80}")
    print("🖊️  轨迹回放工具")
    print(f"{'='*80}")
    print(f"场景: {args.scenario} - {scenario['name']}")
    print(f"可用episodes: {len(episode_files)}")
    print(f"{'='*80}")
    
    # 加载MuJoCo模型
    model = mujoco.MjModel.from_xml_path((get_so100_models_dir() / "scene.xml").as_posix())
    data = mujoco.MjData(model)
    
    if args.quad_view:
        # 四分屏模式：自动选择前4个episodes
        if args.episodes:
            ep_indices = args.episodes[:4]
        else:
            ep_indices = list(range(min(4, len(episode_files))))
        
        # 加载轨迹
        trajectories = []
        for ep_idx in ep_indices:
            if ep_idx < len(episode_files):
                traj = load_trajectory(episode_files[ep_idx])
                trajectories.append(traj)
        
        # 生成输出文件名
        output_file = scenario_dir / f'quad_view_episodes_{"_".join(map(str, ep_indices))}.mp4'
        
        # 执行四分屏录制
        playback_quad_view(model, data, trajectories, scenario, ep_indices, 
                          output_file, args.speed)
    
    elif args.episodes:
        # 加载多个指定episodes
        trajectories = []
        for ep_idx in args.episodes:
            if ep_idx < len(episode_files):
                traj = load_trajectory(episode_files[ep_idx])
                trajectories.append(traj)
            else:
                print(f"⚠️ 警告: Episode {ep_idx} 不存在，跳过")
        
        if args.compare:
            playback_comparison(model, data, trajectories, scenario, args.episodes)
        else:
            # 依次回放
            success_indices = []
            for ep_idx, traj in zip(args.episodes, trajectories):
                print(f"\n播放 Episode {ep_idx}")
                output_file = scenario_dir / f'playback_episode_{ep_idx}.mp4' if args.record else None
                success = playback_single_episode(model, data, traj, scenario, 
                                      args.speed, args.record, output_file)
                if args.save_success_seeds and success:
                    success_indices.append(ep_idx)
                if ep_idx != args.episodes[-1] and not args.save_success_seeds:
                    print("\n按Enter继续下一个episode...")
                    input()
            if args.save_success_seeds:
                seed_file = scenario_dir / 'seed.txt'
                with open(seed_file, 'w') as f:
                    f.write(','.join(str(idx) for idx in success_indices))
                print(f"\n✅ 已保存成功episode索引到: {seed_file}")
    
    elif args.episode is not None:
        # 回放单个episode
        if args.episode >= len(episode_files):
            print(f"❌ 错误: Episode {args.episode} 不存在")
            return
        
        trajectory = load_trajectory(episode_files[args.episode])
        output_file = scenario_dir / f'playback_episode_{args.episode}.mp4' if args.record else None
        playback_single_episode(model, data, trajectory, scenario, 
                              args.speed, args.record, output_file)
    
    else:
        # 回放所有episodes
        print(f"\n将依次回放所有 {len(episode_files)} 个episodes")
        success_indices = []
        for ep_idx, ep_file in enumerate(episode_files):
            print(f"\n{'='*80}")
            print(f"Episode {ep_idx}/{len(episode_files)-1}")
            print(f"{'='*80}")
            
            trajectory = load_trajectory(ep_file)
            output_file = scenario_dir / f'playback_episode_{ep_idx}.mp4' if args.record else None
            success = playback_single_episode(model, data, trajectory, scenario, 
                                  args.speed, args.record, output_file)
            if args.save_success_seeds and success:
                success_indices.append(ep_idx)
            
            if ep_idx < len(episode_files) - 1 and not args.save_success_seeds:
                print("\n按Enter继续下一个episode...")
                input()
        if args.save_success_seeds:
            seed_file = scenario_dir / 'seed.txt'
            with open(seed_file, 'w') as f:
                f.write(','.join(str(idx) for idx in success_indices))
            print(f"\n✅ 已保存成功episode索引到: {seed_file}")
    
    print(f"\n{'='*80}")
    print("✅ 回放完成！")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

