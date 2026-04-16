#!/usr/bin/env python3
"""
使用MPlib运动规划算法为关键帧序列生成多样化轨迹

核心思路：
1. 读取手动标注的关键帧（waypoints）
2. 对每两个关键帧之间，使用MPlib规划生成轨迹段
3. 通过不同的随机种子生成多样化的轨迹
4. 保存为训练数据格式
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import mplib
import mujoco
import numpy as np

from lerobot_sim_lab.utils.paths import get_so100_models_dir

try:
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import util as ou
    HAS_OMPL = True
except Exception:
    ob = None
    og = None
    ou = None
    HAS_OMPL = False

# SO-100机械臂配置
JOINT_NAMES = ['Rotation', 'Pitch', 'Elbow', 'Wrist_Pitch', 'Wrist_Roll', 'Jaw']


def _set_ompl_seed(seed: int) -> bool:
    if seed is None or not HAS_OMPL:
        return False
    if hasattr(ob, "RNG"):
        ob.RNG.setSeed(int(seed))
        return True
    if ou is not None and hasattr(ou, "RNG"):
        ou.RNG.setSeed(int(seed))
        return True
    return False


class MPLibTrajectoryGenerator:
    """使用MPlib为关键帧序列生成多样化轨迹"""
    
    def __init__(self, urdf_path: str, srdf_path: str, scene_xml_path: str, move_group: str = "Moving Jaw"):
        """
        Args:
            urdf_path: 机械臂URDF路径
            srdf_path: SRDF配置文件路径（定义planning group等）
            scene_xml_path: MuJoCo场景XML路径（用于可视化）
            move_group: 末端执行器链接名称（注意：mplib会将"Moving_Jaw"解析为"Moving Jaw"，带空格）
        """
        # 保存配置（用于重新初始化planner）
        self.urdf_path = urdf_path
        self.srdf_path = srdf_path
        self.move_group = move_group
        
        # 初始化MPlib规划器
        self.planner = mplib.Planner(
            urdf=urdf_path,
            srdf=srdf_path,
            move_group=move_group,
            verbose=False
        )
        
        # 加载MuJoCo场景（用于可视化）
        self.mj_model = mujoco.MjModel.from_xml_path(scene_xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.joint_limits = self._get_joint_limits()
        self._ompl_seeded = False
        
        print("✅ MPlib规划器已初始化")
        print(f"   URDF: {urdf_path}")
        print(f"   Move Group: {move_group}")
        print(f"   DOF: {len(self.planner.move_group_joint_indices)}")
    
    def reinitialize_planner(self):
        """重新初始化planner（重置内部随机状态）"""
        self.planner = mplib.Planner(
            urdf=self.urdf_path,
            srdf=self.srdf_path,
            move_group=self.move_group,
            verbose=False
        )
    
    def add_collision_objects(self, scenario_config: dict):
        """
        将场景中的物体添加到MPlib的碰撞检测环境
        
        注意：当前版本简化处理，仅依赖自碰撞检测
        TODO: 根据MPlib版本添加环境碰撞物体
        
        Args:
            scenario_config: 场景配置（包含笔、盒子等物体的位置）
        """
        # 注意：MPlib的环境碰撞添加API取决于版本
        # 当前先使用自碰撞检测，避免与环境物体碰撞可以通过：
        # 1. 确保关键帧已经避障
        # 2. 使用较小的RRT采样范围
        
        print("✅ 规划器已配置（使用自碰撞检测）")
        print("   注意：请确保手动标注的关键帧已避开环境障碍物")
    
    def _get_joint_limits(self):
        limits = []
        for name in JOINT_NAMES:
            joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            low, high = self.mj_model.jnt_range[joint_id]
            limits.append((float(low), float(high)))
        return limits
    
    
    def plan_segment(self, 
                     start_qpos: np.ndarray, 
                     goal_qpos: np.ndarray,
                     time_step: float = 0.01,
                     use_rrt: bool = True,
                     use_ompl: bool = False,
                     ompl_planner: str = "RRTConnect",
                     ompl_simplify: bool = True,
                     random_seed: int = None,
                     check_collision: bool = True,
                     rrt_range: float = 0.1,
                     planning_time: float = 2.0,
                     via_point_noise: float = 0.0) -> np.ndarray:
        """
        使用MPlib规划从start到goal的轨迹段
        
        Args:
            start_qpos: 起始关节角度 [6]
            goal_qpos: 目标关节角度 [6]
            time_step: 时间步长（秒），默认0.01s=100Hz，越小越流畅
            use_rrt: 是否使用RRT规划（True=多样化，False=直接插值）
            random_seed: 随机种子（不同种子产生不同路径）
            check_collision: 是否检查自碰撞（可选，SRDF配置不当时会误报）
            rrt_range: RRT采样范围（越大越多样化，但可能更慢）
            planning_time: 最大规划时间（秒）
            via_point_noise: 中间路径点扰动大小（弧度），>0时会添加随机中间点
        
        Returns:
            trajectory: [T, 6] 轨迹数组
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            if hasattr(self.planner, "set_seed"):
                self.planner.set_seed(random_seed)
            elif hasattr(self.planner, "set_random_seed"):
                self.planner.set_random_seed(random_seed)
            elif hasattr(mplib, "set_seed"):
                mplib.set_seed(random_seed)
            elif hasattr(mplib, "set_random_seed"):
                mplib.set_random_seed(random_seed)
        
        # 如果使用via point，则分两段规划：start → via_point → goal
        if via_point_noise > 0 and (use_rrt or use_ompl):
            print(f"    💫 使用via point增加路径多样性（扰动: ±{via_point_noise:.3f} rad）")
            
            # 生成中间路径点：在起点和终点之间插值，并添加随机扰动
            via_point = (start_qpos + goal_qpos) / 2.0
            noise = np.random.uniform(-via_point_noise, via_point_noise, size=6)
            via_point = via_point + noise
            
            # 检查via point是否有自碰撞
            if self.planner.check_for_self_collision(via_point):
                print("    ⚠️  随机生成的via point存在自碰撞，减小扰动重试")
                # 减小扰动再试一次
                via_point = (start_qpos + goal_qpos) / 2.0 + noise * 0.5
            
            # 分两段规划
            print("      第1段: start → via_point")
            seg1 = self.plan_segment(
                start_qpos=start_qpos,
                goal_qpos=via_point,
                time_step=time_step,
                use_rrt=use_rrt,
                use_ompl=use_ompl,
                ompl_planner=ompl_planner,
                ompl_simplify=ompl_simplify,
                random_seed=random_seed,
                check_collision=check_collision,
                rrt_range=rrt_range,
                planning_time=planning_time,
                via_point_noise=0  # 递归时不再添加via point
            )
            
            print("      第2段: via_point → goal")
            seg2 = self.plan_segment(
                start_qpos=via_point,
                goal_qpos=goal_qpos,
                time_step=time_step,
                use_rrt=use_rrt,
                use_ompl=use_ompl,
                ompl_planner=ompl_planner,
                ompl_simplify=ompl_simplify,
                random_seed=random_seed + 1000 if random_seed else None,
                check_collision=check_collision,
                rrt_range=rrt_range,
                planning_time=planning_time,
                via_point_noise=0  # 递归时不再添加via point
            )
            
            # 合并两段（跳过重复点）
            trajectory = np.concatenate([seg1, seg2[1:]], axis=0)
            print(f"    ✅ 通过via point规划完成: {len(trajectory)} 步")
            return trajectory
        
        # 检查起始和目标姿态是否存在自碰撞
        start_collision = self.planner.check_for_self_collision(start_qpos)
        goal_collision = self.planner.check_for_self_collision(goal_qpos)
        
        if start_collision:
            print("  ⚠️ 警告: 起始姿态存在自碰撞！")
        if goal_collision:
            print("  ⚠️ 警告: 目标姿态存在自碰撞！")
        
        # 如果不使用碰撞检测，或者姿态存在碰撞问题，直接使用线性插值
        if not check_collision or start_collision or goal_collision:
            if not check_collision:
                print("  ℹ️  已禁用碰撞检测，使用线性插值")
            else:
                print("  ⚠️  检测到碰撞问题，使用线性插值")
            return self._linear_interpolate(start_qpos, goal_qpos, int(np.linalg.norm(goal_qpos - start_qpos) / 0.01 / time_step))
        
        if use_ompl:
            trajectory = self._plan_segment_ompl(
                start_qpos=start_qpos,
                goal_qpos=goal_qpos,
                time_step=time_step,
                random_seed=random_seed,
                check_collision=check_collision,
                planning_time=planning_time,
                ompl_planner=ompl_planner,
                ompl_simplify=ompl_simplify
            )
            if trajectory is not None:
                return trajectory
            return self._linear_interpolate(start_qpos, goal_qpos, 50)
        if use_rrt:
            # 使用RRT规划（具有随机性）
            try:
                # 注意：plan_qpos 的 goal_qposes 参数是列表！
                # simplify=False 保留RRT原始路径，避免被简化成相同的直线
                result = self.planner.plan_qpos(
                    goal_qposes=[goal_qpos],  # 必须是列表
                    current_qpos=start_qpos,
                    time_step=time_step,
                    rrt_range=rrt_range,  # RRT采样范围，影响路径多样性
                    planning_time=planning_time,  # 最大规划时间（秒）
                    simplify=False,  # 🔥 关键！不简化路径，保留RRT的多样性
                    verbose=True  # 打开详细日志
                )
                
                if result['status'] == "Success":
                    trajectory = np.array(result['position'])  # [T, 6]
                    print(f"  ✅ RRT规划成功: {len(trajectory)} 步")
                    return trajectory
                else:
                    print(f"  ❌ RRT规划失败: {result['status']}")
                    print(f"     原因: {result.get('message', 'Unknown')}")
                    print("     ⚠️  回退到线性插值")
                    return self._linear_interpolate(start_qpos, goal_qpos, 50)
            except Exception as e:
                print(f"  ❌ RRT规划异常: {e}")
                print("     回退到线性插值")
                return self._linear_interpolate(start_qpos, goal_qpos, 50)
        else:
            # 直接使用线性插值（用于对比）
            return self._linear_interpolate(start_qpos, goal_qpos, 50)
    
    def _make_ompl_planner(self, planner_name: str, si):
        planners = {
            "RRTConnect": og.RRTConnect,
            "RRT": og.RRT,
            "RRTstar": og.RRTstar,
            "PRM": og.PRM,
            "PRMstar": og.PRMstar,
            "KPIECE1": og.KPIECE1,
            "EST": og.EST,
            "BiEST": og.BiEST,
            "BITstar": og.BITstar,
            "FMT": og.FMT,
            "RRTsharp": og.RRTsharp,
        }
        planner_cls = planners.get(planner_name, og.RRTConnect)
        return planner_cls(si)
    
    def _plan_segment_ompl(self,
                           start_qpos: np.ndarray,
                           goal_qpos: np.ndarray,
                           time_step: float,
                           random_seed: int,
                           check_collision: bool,
                           planning_time: float,
                           ompl_planner: str,
                           ompl_simplify: bool) -> np.ndarray | None:
        if not HAS_OMPL:
            raise RuntimeError("OMPL not available")
        dim = len(start_qpos)
        space = ob.RealVectorStateSpace(dim)
        bounds = ob.RealVectorBounds(dim)
        for i, (low, high) in enumerate(self.joint_limits):
            bounds.setLow(i, low)
            bounds.setHigh(i, high)
        space.setBounds(bounds)
        if random_seed is not None and not self._ompl_seeded:
            if hasattr(ob, "RNG"):
                ob.RNG.setSeed(int(random_seed))
                self._ompl_seeded = True
            elif ou is not None and hasattr(ou, "RNG"):
                ou.RNG.setSeed(int(random_seed))
                self._ompl_seeded = True
        si = ob.SpaceInformation(space)
        if check_collision:
            def is_valid(state):
                qpos = np.array([state[i] for i in range(dim)], dtype=float)
                return not self.planner.check_for_self_collision(qpos)
        else:
            def is_valid(state):
                return True
        si.setStateValidityChecker(ob.StateValidityCheckerFn(is_valid))
        si.setup()
        start = ob.State(space)
        goal = ob.State(space)
        for i in range(dim):
            start[i] = float(start_qpos[i])
            goal[i] = float(goal_qpos[i])
        pdef = ob.ProblemDefinition(si)
        pdef.setStartAndGoalStates(start, goal)
        planner = self._make_ompl_planner(ompl_planner, si)
        planner.setProblemDefinition(pdef)
        planner.setup()
        solved = planner.solve(float(planning_time))
        if not solved:
            return None
        path = pdef.getSolutionPath()
        if ompl_simplify:
            simplifier = og.PathSimplifier(si)
            simplifier.simplifyMax(path)
            simplifier.smoothBSpline(path)
        steps = max(2, int(np.linalg.norm(goal_qpos - start_qpos) / time_step))
        path.interpolate(steps)
        states = path.getStates()
        trajectory = np.array([[state[i] for i in range(dim)] for state in states], dtype=float)
        return trajectory
    
    def _linear_interpolate(self, start: np.ndarray, goal: np.ndarray, steps: int) -> np.ndarray:
        """线性插值（作为备选方案）"""
        trajectory = []
        for i in range(steps):
            alpha = i / steps
            config = start + alpha * (goal - start)
            trajectory.append(config)
        trajectory.append(goal)
        return np.array(trajectory)
    
    def generate_diverse_trajectories(self,
                                       waypoints: list,
                                       num_episodes: int = 10,
                                       base_seed: int = 42,
                                       time_step: float = 0.01,
                                       rrt_range: float = 0.1,
                                       planning_time: float = 2.0,
                                      use_collision: bool = True,
                                      waypoint_noise: float = 0.0,
                                      use_ompl: bool = False,
                                      ompl_planner: str = "RRTConnect",
                                      ompl_simplify: bool = True,
                                      keyframe_noise: float = 0.0) -> list:
        """
        为相同的关键帧序列生成多样化的轨迹
        
        Args:
            waypoints: 关键帧列表 [{name, config, steps}, ...]
            num_episodes: 生成的轨迹数量
            base_seed: 基础随机种子
            time_step: 时间步长（秒），默认0.01s=100Hz，越小越流畅
            rrt_range: RRT采样范围（越大越多样化）
            planning_time: 最大规划时间（秒）
            use_collision: 是否启用碰撞检测
            waypoint_noise: 中间路径点扰动大小（弧度），>0时增加路径多样性
        
        Returns:
            episodes: 列表，每个元素是一条完整轨迹 [num_episodes] 个 [T, 6] 数组
        """
        episodes = []
        
        print(f"\n{'='*80}")
        print(f"开始生成 {num_episodes} 条多样化轨迹")
        print(f"关键帧数量: {len(waypoints)}")
        print(f"{'='*80}\n")
        
        for ep_idx in range(num_episodes):
            print(f"Episode {ep_idx + 1}/{num_episodes}:")
            
            # 重新初始化planner（重置C++内部随机状态）
            if ep_idx > 0:
                print("  🔄 重新初始化planner以获得不同的随机状态...")
                self.reinitialize_planner()
            
            trajectory = []
            
            rng = np.random.default_rng(base_seed + ep_idx * 1000 + 123)
            episode_waypoints = []
            for wp in waypoints:
                config = np.array(wp['config'])
                if keyframe_noise > 0.0 and wp.get('perturb', False):
                    noise = rng.uniform(-keyframe_noise, keyframe_noise, size=config.shape)
                    noise[-1] = 0.0
                    config = config + noise
                    for idx, (low, high) in enumerate(self.joint_limits):
                        config[idx] = float(np.clip(config[idx], low, high))
                episode_waypoints.append({**wp, 'config': config.tolist()})
            
            # 对每两个关键帧之间进行规划
            for i in range(len(waypoints) - 1):
                start_config = np.array(episode_waypoints[i]['config'])
                goal_config = np.array(episode_waypoints[i+1]['config'])
                
                # 关键：使用不同的随机种子生成不同的路径
                segment_seed = base_seed + ep_idx * 1000 + i
                
                print(f"  规划段 {i+1}/{len(waypoints)-1}: {episode_waypoints[i]['name']} → {episode_waypoints[i+1]['name']}")
                segment = self.plan_segment(
                    start_qpos=start_config,
                    goal_qpos=goal_config,
                    time_step=time_step,
                    random_seed=segment_seed,
                    rrt_range=rrt_range,
                    planning_time=planning_time,
                    check_collision=use_collision,
                    via_point_noise=waypoint_noise,
                    use_ompl=use_ompl,
                    ompl_planner=ompl_planner,
                    ompl_simplify=ompl_simplify
                )
                
                # 拼接轨迹段（避免重复点）
                if i == 0:
                    trajectory.append(segment)
                else:
                    trajectory.append(segment[1:])  # 跳过第一个点（已经是前一段的最后一个点）
            
            # 合并所有段
            full_trajectory = np.concatenate(trajectory, axis=0)
            episodes.append(full_trajectory)
            
            print(f"  ✅ Episode {ep_idx + 1} 生成完成: {len(full_trajectory)} 步\n")
        
        print(f"{'='*80}")
        print(f"✅ 所有 {num_episodes} 条轨迹生成完成！")
        print(f"{'='*80}\n")
        
        return episodes
    
    def save_episodes(self, episodes: list, scenario_dir: Path, scenario_id: int, episode_offset: int = 0, write_summary: bool = True):
        """保存轨迹数据"""
        output_dir = scenario_dir / 'trajectories'
        output_dir.mkdir(exist_ok=True)
        
        for ep_idx, trajectory in enumerate(episodes):
            output_file = output_dir / f'episode_{ep_idx + episode_offset:03d}.npz'
            np.savez(
                output_file,
                trajectory=trajectory,  # [T, 6]
                joint_names=JOINT_NAMES,
                scenario_id=scenario_id,
            )
            print(f"  保存: {output_file}")
        
        if write_summary:
            # 保存汇总信息
            summary = {
                'scenario_id': scenario_id,
                'num_episodes': len(episodes),
                'trajectory_lengths': [len(traj) for traj in episodes],
                'total_steps': sum(len(traj) for traj in episodes),
            }
            with open(output_dir / 'summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\n✅ 所有数据已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="使用MPlib生成多样化轨迹")
    parser.add_argument('--scenario', type=int, required=True, choices=[0,1,2,3,4,5],
                       help="场景ID")
    parser.add_argument('--num-episodes', type=int, default=10,
                       help="每个场景生成的episode数量")
    parser.add_argument('--seed', type=int, default=42,
                       help="基础随机种子")
    parser.add_argument('--time-step', type=float, default=0.02,
                       help="时间步长（秒），默认0.01s=100Hz，越小越流畅（推荐：0.01-0.033）")
    parser.add_argument('--rrt-range', type=float, default=0.1,
                       help="RRT采样范围，默认0.1，越大越多样化但速度慢")
    parser.add_argument('--planning-time', type=float, default=2.0,
                       help="最大规划时间（秒），默认2.0")
    parser.add_argument('--planner-backend', type=str, default="mplib", choices=["mplib", "ompl"],
                       help="规划器后端")
    parser.add_argument('--ompl-planner', type=str, default="RRTConnect",
                       help="OMPL planner名称")
    parser.add_argument('--ompl-no-simplify', action='store_true',
                       help="禁用OMPL路径简化")
    parser.add_argument('--ompl-subprocess', action='store_true',
                       help="OMPL每个episode使用子进程生成")
    parser.add_argument('--ompl-subprocess-worker', action='store_true',
                       help=argparse.SUPPRESS)
    parser.add_argument('--episode-offset', type=int, default=0,
                       help=argparse.SUPPRESS)
    parser.add_argument('--no-collision', action='store_true',
                       help="禁用碰撞检测（用于调试SRDF配置问题）")
    parser.add_argument('--via-point-noise', type=float, default=0.0,
                       help="中间路径点扰动大小（弧度），默认0.1，推荐0.05-0.3，设为0则不使用via points")
    parser.add_argument('--keyframe-noise', type=float, default=0.0,
                       help="关键帧扰动大小（弧度），仅对perturb=true的关键帧生效")
    args = parser.parse_args()
    
    # 场景目录
    scenario_dir = Path(f'pen_grab_tuning/scenario_{args.scenario}')
    waypoints_file = scenario_dir / 'waypoints.json'
    
    if not waypoints_file.exists():
        print(f"❌ 错误: {waypoints_file} 不存在")
        print("   请先使用 python -m lerobot_sim_lab.tuning.tune_pen_grab_multi --mode control 生成关键帧")
        return
    
    # 加载关键帧
    with open(waypoints_file) as f:
        waypoints_data = json.load(f)
    
    if len(waypoints_data['waypoints']) < 2:
        print("❌ 错误: 至少需要2个关键帧才能生成轨迹")
        return
    
    print(f"\n{'='*80}")
    print("🖊️  MPlib 多样化轨迹生成器")
    print(f"{'='*80}")
    print(f"场景: {args.scenario}")
    print(f"关键帧数量: {len(waypoints_data['waypoints'])}")
    print(f"目标episode数量: {args.num_episodes}")
    print(f"随机种子: {args.seed}")
    print(f"时间步长: {args.time_step}s ({1/args.time_step:.0f}Hz)")
    print(f"RRT采样范围: {args.rrt_range}")
    print(f"规划超时: {args.planning_time}s")
    print(f"Via point扰动: {args.via_point_noise} rad ({np.degrees(args.via_point_noise):.1f}°)")
    print(f"规划后端: {args.planner_backend}")
    print(f"OMPL planner: {args.ompl_planner}")
    print(f"{'='*80}\n")
    
    # OMPL子进程生成
    if args.planner_backend == "ompl" and args.ompl_subprocess and not args.ompl_subprocess_worker and args.num_episodes > 1:
        script_path = str(Path(__file__).resolve())
        for ep_idx in range(args.num_episodes):
            seed = args.seed + ep_idx
            cmd = [
                sys.executable, script_path,
                '--scenario', str(args.scenario),
                '--num-episodes', '1',
                '--seed', str(seed),
                '--time-step', str(args.time_step),
                '--rrt-range', str(args.rrt_range),
                '--planning-time', str(args.planning_time),
                '--via-point-noise', str(args.via_point_noise),
                '--planner-backend', 'ompl',
                '--ompl-planner', args.ompl_planner,
                '--episode-offset', str(ep_idx),
                '--ompl-subprocess-worker',
            ]
            if args.ompl_no_simplify:
                cmd.append('--ompl-no-simplify')
            if args.no_collision:
                cmd.append('--no-collision')
            subprocess.run(cmd, check=True)
        output_dir = Path(f'pen_grab_tuning/scenario_{args.scenario}') / 'mplib_trajectories'
        trajectories = []
        for ep_idx in range(args.num_episodes):
            data = np.load(output_dir / f'episode_{ep_idx:03d}.npz')
            trajectories.append(data['trajectory'])
        summary = {
            'scenario_id': args.scenario,
            'num_episodes': args.num_episodes,
            'trajectory_lengths': [len(traj) for traj in trajectories],
            'total_steps': sum(len(traj) for traj in trajectories),
        }
        with open(output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n✅ 所有数据已保存到: {output_dir}")
        return

    ompl_seeded = False
    if args.planner_backend == "ompl":
        ompl_seeded = _set_ompl_seed(args.seed)

    # 初始化MPlib生成器
    so100_dir = get_so100_models_dir()
    generator = MPLibTrajectoryGenerator(
        urdf_path=(so100_dir / 'so100.urdf').as_posix(),
        srdf_path=(so100_dir / 'so100_mplib.srdf').as_posix(),
        scene_xml_path=(so100_dir / 'scene.xml').as_posix(),
        move_group='Moving Jaw'  # 注意：mplib将"Moving_Jaw"解析为带空格的"Moving Jaw"
    )
    if ompl_seeded:
        generator._ompl_seeded = True
    
    # 添加场景物体到碰撞环境
    generator.add_collision_objects(waypoints_data)
    
    # 生成多样化轨迹
    episodes = generator.generate_diverse_trajectories(
        waypoints=waypoints_data['waypoints'],
        num_episodes=args.num_episodes,
        base_seed=args.seed,
        time_step=args.time_step,
        rrt_range=args.rrt_range,
        planning_time=args.planning_time,
        use_collision=not args.no_collision,
        waypoint_noise=args.via_point_noise,
        use_ompl=args.planner_backend == "ompl",
        ompl_planner=args.ompl_planner,
        ompl_simplify=not args.ompl_no_simplify,
        keyframe_noise=args.keyframe_noise
    )
    
    # 保存数据
    generator.save_episodes(
        episodes,
        scenario_dir,
        args.scenario,
        episode_offset=args.episode_offset,
        write_summary=not args.ompl_subprocess_worker
    )
    
    # 显示轨迹多样性统计
    print(f"\n{'='*80}")
    print("📊 轨迹多样性统计")
    print(f"{'='*80}")
    lengths = [len(traj) for traj in episodes]
    print(f"轨迹长度范围: {min(lengths)} - {max(lengths)} 步")
    print(f"平均长度: {np.mean(lengths):.1f} ± {np.std(lengths):.1f} 步")
    print(f"总步数: {sum(lengths)} 步")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
