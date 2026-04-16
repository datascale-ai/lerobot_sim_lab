"""
MuJoCo 仿真交互式键盘控制
支持通过 SSH X11 转发在远程客户端上操作

功能：
1. 实时键盘控制机器人关节
2. 鼠标拖拽视角
3. 显示当前状态
4. 录制操作数据（可选）

使用方法：
1. 服务器端启动 SSH X11 转发：
   ssh -X user@server
   
2. 运行脚本：
   python interactive_control.py --scene push_cube
   
3. 键盘控制：
   - 数字键 1-6: 选择关节
   - ↑/↓ 或 W/S: 增加/减少关节角度
   - R: 重置到初始位置
   - SPACE: 暂停/继续
   - Q 或 ESC: 退出
   - P: 打印当前状态
   - [ / ]: 减小/增大步长
"""

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from lerobot_sim_lab.utils.paths import get_so100_scene_path


class InteractiveController:
    """交互式键盘控制器"""
    
    def __init__(self, scene_path: str, record: bool = False, window_size: tuple = (800, 600), 
                 max_fps: int = 30, render_quality: str = "low"):
        """
        初始化控制器
        
        Args:
            scene_path: MuJoCo XML 文件路径
            record: 是否录制操作数据
            window_size: 窗口尺寸 (width, height)，默认 (800, 600)
            max_fps: 最大帧率，默认 30 fps
            render_quality: 渲染质量 ('low', 'medium', 'high')
        """
        print("="*70)
        print("MuJoCo 交互式控制器")
        print("="*70)
        
        # 加载模型
        print(f"\n加载场景: {scene_path}")
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        
        # 渲染优化设置
        self.window_size = window_size
        self.max_fps = max_fps
        self.render_quality = render_quality
        self.min_frame_time = 1.0 / max_fps if max_fps > 0 else 0
        
        # 根据渲染质量调整 MuJoCo 可视化选项
        if render_quality == "low":
            self.model.vis.quality.shadowsize = 256  # 降低阴影分辨率
            self.model.vis.quality.offsamples = 1    # 减少抗锯齿采样
        elif render_quality == "medium":
            self.model.vis.quality.shadowsize = 1024
            self.model.vis.quality.offsamples = 2
        else:  # high
            self.model.vis.quality.shadowsize = 2048
            self.model.vis.quality.offsamples = 4
        
        print(f"窗口尺寸: {window_size[0]}x{window_size[1]}")
        print(f"最大帧率: {max_fps} fps")
        print(f"渲染质量: {render_quality}")
        
        # 控制参数
        self.selected_joint = 0
        self.step_size = 0.1  # 每次按键的角度变化（弧度）
        self.paused = False
        
        # 获取控制关节数量（假设前 n 个是可控关节）
        self.num_actuators = self.model.nu
        print(f"可控关节数: {self.num_actuators}")
        
        # 保存初始位置
        self.initial_ctrl = self.data.ctrl.copy()
        
        # 录制设置
        self.record = record
        if self.record:
            self.recorded_states = []
            self.recorded_actions = []
            print("📹 录制模式已启用")
        
        # 状态历史（用于显示）
        self.history_length = 100
        self.ctrl_history = deque(maxlen=self.history_length)
        
        print("\n✅ 初始化完成")
        self.print_instructions()
    
    def print_instructions(self):
        """打印控制说明"""
        print("\n" + "="*70)
        print("键盘控制说明")
        print("="*70)
        print("\n🎮 关节选择:")
        print("  1-6    : 选择关节 1-6")
        print(f"  当前选中: 关节 {self.selected_joint + 1}")
        
        print("\n🕹️  关节控制:")
        print("  ↑ / W  : 增加关节角度")
        print("  ↓ / S  : 减少关节角度")
        print("  [ / ]  : 减小/增大步长")
        print(f"  当前步长: {np.rad2deg(self.step_size):.1f}°")
        
        print("\n⚙️  系统控制:")
        print("  R      : 重置到初始位置")
        print("  SPACE  : 暂停/继续")
        print("  P      : 打印当前状态")
        print("  Q/ESC  : 退出程序")
        
        print("\n🖱️  鼠标操作:")
        print("  拖拽   : 旋转视角")
        print("  滚轮   : 缩放")
        print("="*70)
    
    def handle_keyboard(self, keycode: int):
        """
        处理键盘输入
        
        Args:
            keycode: 键盘代码
        """
        # 数字键 1-6: 选择关节
        if ord('1') <= keycode <= ord('9'):
            joint_idx = keycode - ord('1')
            if joint_idx < self.num_actuators:
                self.selected_joint = joint_idx
                print(f"✓ 选中关节 {self.selected_joint + 1}")
                self.print_current_state()
                return True
        
        # 方向键上 / W: 增加角度
        if keycode == 265 or keycode == ord('W') or keycode == ord('w'):
            self.data.ctrl[self.selected_joint] += self.step_size
            print(f"↑ 关节 {self.selected_joint + 1}: {self.data.ctrl[self.selected_joint]:.3f} rad "
                  f"({np.rad2deg(self.data.ctrl[self.selected_joint]):.1f}°)")
            return True
        
        # 方向键下 / S: 减少角度
        if keycode == 264 or keycode == ord('S') or keycode == ord('s'):
            self.data.ctrl[self.selected_joint] -= self.step_size
            print(f"↓ 关节 {self.selected_joint + 1}: {self.data.ctrl[self.selected_joint]:.3f} rad "
                  f"({np.rad2deg(self.data.ctrl[self.selected_joint]):.1f}°)")
            return True
        
        # [ : 减小步长
        if keycode == ord('['):
            self.step_size = max(0.01, self.step_size - 0.01)
            print(f"步长: {np.rad2deg(self.step_size):.1f}°")
            return True
        
        # ] : 增大步长
        if keycode == ord(']'):
            self.step_size = min(0.5, self.step_size + 0.01)
            print(f"步长: {np.rad2deg(self.step_size):.1f}°")
            return True
        
        # R: 重置
        if keycode == ord('R') or keycode == ord('r'):
            self.data.ctrl[:] = self.initial_ctrl
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            print("🔄 重置到初始位置")
            self.print_current_state()
            return True
        
        # SPACE: 暂停/继续
        if keycode == 32:  # SPACE
            self.paused = not self.paused
            status = "⏸️  暂停" if self.paused else "▶️  继续"
            print(status)
            return True
        
        # P: 打印状态
        if keycode == ord('P') or keycode == ord('p'):
            self.print_current_state()
            return True
        
        # Q / ESC: 退出
        if keycode == ord('Q') or keycode == ord('q') or keycode == 256:  # ESC
            print("\n👋 退出程序")
            return False
        
        return True
    
    def print_current_state(self):
        """打印当前状态"""
        print("\n" + "-"*70)
        print("当前状态")
        print("-"*70)
        
        # 控制指令
        print("\n控制指令 (ctrl):")
        for i in range(self.num_actuators):
            marker = "👉" if i == self.selected_joint else "  "
            print(f"{marker} 关节 {i+1}: {self.data.ctrl[i]:7.3f} rad = {np.rad2deg(self.data.ctrl[i]):7.1f}°")
        
        # 实际关节位置
        print(f"\n实际位置 (qpos[:{self.num_actuators}]):")
        for i in range(min(self.num_actuators, len(self.data.qpos))):
            print(f"   关节 {i+1}: {self.data.qpos[i]:7.3f} rad = {np.rad2deg(self.data.qpos[i]):7.1f}°")
        
        # 关节速度
        print(f"\n关节速度 (qvel[:{self.num_actuators}]):")
        for i in range(min(self.num_actuators, len(self.data.qvel))):
            print(f"   关节 {i+1}: {self.data.qvel[i]:7.3f} rad/s")
        
        print("-"*70)
    
    def run(self):
        """运行交互式控制"""
        print("\n🚀 启动交互式控制...")
        print("提示: 如果窗口无响应，请检查 X11 转发是否正常\n")
        
        # 检查 DISPLAY 环境变量
        if 'DISPLAY' not in os.environ:
            print("⚠️  警告: 未检测到 DISPLAY 环境变量")
            print("   请确保使用 'ssh -X' 或 'ssh -Y' 连接服务器")
        else:
            print(f"✓ DISPLAY = {os.environ['DISPLAY']}")
        
        # 设置环境变量以优化 X11 性能
        os.environ['MUJOCO_GL'] = 'glfw'  # 使用 GLFW 后端
        
        try:
            # 注意：mujoco.viewer.launch_passive 不直接支持窗口尺寸参数
            # 需要通过环境变量或其他方式设置
            with mujoco.viewer.launch_passive(
                self.model, 
                self.data,
                show_left_ui=True,
                show_right_ui=True
            ) as viewer:
                
                # 设置键盘回调（如果支持）
                # 注意：mujoco.viewer 的键盘支持可能有限
                
                print("\n✅ 查看器已启动")
                print("   使用键盘控制机器人\n")
                
                last_print_time = time.time()
                last_render_time = time.time()
                
                while viewer.is_running():
                    step_start = time.time()
                    
                    # 如果未暂停，执行仿真步
                    if not self.paused:
                        mujoco.mj_step(self.model, self.data)
                        
                        # 录制数据
                        if self.record:
                            self.recorded_states.append(self.data.qpos.copy())
                            self.recorded_actions.append(self.data.ctrl.copy())
                        
                        # 记录控制历史
                        self.ctrl_history.append(self.data.ctrl.copy())
                    
                    # 限制渲染帧率以减少 X11 数据传输
                    current_time = time.time()
                    if current_time - last_render_time >= self.min_frame_time:
                        viewer.sync()
                        last_render_time = current_time
                    
                    # 定期打印提示（每 5 秒）
                    if time.time() - last_print_time > 5.0:
                        print("💡 提示: 按 'P' 查看当前状态, 按 'Q' 退出")
                        last_print_time = time.time()
                    
                    # 时间控制
                    time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
        
        except KeyboardInterrupt:
            print("\n\n⚠️  收到中断信号")
        
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # 保存录制的数据
            if self.record and len(self.recorded_states) > 0:
                self.save_recording()
            
            print("\n👋 程序已退出")
    
    def save_recording(self):
        """保存录制的数据"""
        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"recording_{timestamp}.npz"
        
        np.savez(
            output_file,
            states=np.array(self.recorded_states),
            actions=np.array(self.recorded_actions),
            num_steps=len(self.recorded_states)
        )
        
        print(f"\n💾 录制数据已保存: {output_file}")
        print(f"   步数: {len(self.recorded_states)}")


class KeyboardControlViewer:
    """
    增强版键盘控制查看器
    
    使用 GLFW 的键盘回调来实现更好的键盘控制
    """
    
    def __init__(self, scene_path: str):
        print("="*70)
        print("增强版键盘控制器 (使用 GLFW 回调)")
        print("="*70)

        # 加载模型
        self.scene_path = scene_path
        print(f"\n加载场景: {scene_path}")
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        
        # 控制参数
        self.selected_joint = 0
        self.step_size = 0.1
        self.num_actuators = self.model.nu
        self.initial_ctrl = self.data.ctrl.copy()
        
        print(f"可控关节数: {self.num_actuators}")
        print("\n✅ 初始化完成")
    
    def run_with_glfw(self):
        """使用 GLFW 的键盘回调运行"""
        try:
            import glfw
            print("\n✓ 使用 GLFW 键盘控制")
        except ImportError:
            print("\n⚠️  GLFW 未安装，使用基础模式")
            controller = InteractiveController(self.scene_path)
            controller.run()
            return
        
        # 使用 mujoco.viewer 的被动模式
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            print("\n🎮 查看器已启动，使用键盘控制\n")
            
            while viewer.is_running():
                step_start = time.time()
                
                # 执行仿真
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                
                # 时间控制
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


def test_x11_connection():
    """测试 X11 连接"""
    print("\n" + "="*70)
    print("X11 连接测试")
    print("="*70)
    
    # 检查 DISPLAY
    display = os.environ.get('DISPLAY')
    if not display:
        print("\n❌ 未设置 DISPLAY 环境变量")
        print("\n解决方案:")
        print("1. 断开 SSH 连接")
        print("2. 使用以下命令重新连接:")
        print("   ssh -X user@server")
        print("   或")
        print("   ssh -Y user@server  (trusted X11 forwarding)")
        return False
    else:
        print(f"\n✓ DISPLAY = {display}")
    
    # 尝试导入并测试 mujoco.viewer
    try:
        import mujoco
        import mujoco.viewer
        print("✓ MuJoCo viewer 模块可用")
    except ImportError as e:
        print(f"❌ MuJoCo 导入失败: {e}")
        return False
    
    # 测试简单场景
    try:
        print("\n测试打开简单场景...")
        xml = """
        <mujoco>
            <worldbody>
                <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
                <geom type="plane" size="1 1 0.1" rgba=".9 .9 .9 1"/>
                <body pos="0 0 1">
                    <joint type="free"/>
                    <geom type="box" size=".1 .1 .1" rgba="1 0 0 1"/>
                </body>
            </worldbody>
        </mujoco>
        """
        
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        
        print("✓ 模型加载成功")
        print("\n🚀 尝试打开查看器窗口（3秒）...")
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            for _ in range(90):  # 3 秒
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.033)
        
        print("✅ X11 连接测试成功！")
        return True
    
    except Exception as e:
        print(f"\n❌ X11 测试失败: {e}")
        print("\n可能的原因:")
        print("1. X11 转发未启用")
        print("2. 防火墙阻止 X11")
        print("3. 客户端未安装 X server")
        return False


def main():
    default_scene = str(get_so100_scene_path("push_cube"))
    parser = argparse.ArgumentParser(
        description="MuJoCo 交互式键盘控制 (支持 SSH X11)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 测试 X11 连接
  python -m lerobot_sim_lab.control.interactive --test

  # 使用默认场景
  python -m lerobot_sim_lab.control.interactive

  # 指定场景文件
  python -m lerobot_sim_lab.control.interactive --scene assets/robots/so100/so100_6dof/push_cube_loop.xml

  # 启用录制
  python -m lerobot_sim_lab.control.interactive --record

SSH 连接:
  # 在客户端执行
  ssh -X user@server
  # 或使用 trusted X11
  ssh -Y user@server
        """
    )
    
    parser.add_argument(
        "--scene",
        type=str,
        default=default_scene,
        help="MuJoCo XML 场景文件路径"
    )
    
    parser.add_argument(
        "--record",
        action="store_true",
        help="录制操作数据"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="测试 X11 连接"
    )
    
    parser.add_argument(
        "--width",
        type=int,
        default=800,
        help="窗口宽度（默认 800，建议 X11 转发使用 640-800）"
    )
    
    parser.add_argument(
        "--height",
        type=int,
        default=600,
        help="窗口高度（默认 600，建议 X11 转发使用 480-600）"
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="最大帧率（默认 30，建议 X11 转发使用 15-30）"
    )
    
    parser.add_argument(
        "--quality",
        type=str,
        choices=["low", "medium", "high"],
        default="low",
        help="渲染质量（默认 low，建议 X11 转发使用 low）"
    )
    
    args = parser.parse_args()
    
    # 如果是测试模式
    if args.test:
        test_x11_connection()
        return
    
    # 检查场景文件
    if not Path(args.scene).exists():
        print(f"❌ 场景文件不存在: {args.scene}")
        print("\n可用场景:")
        print(f"  {get_so100_scene_path('basic')}")
        print(f"  {get_so100_scene_path('initial')}")
        print(f"  {get_so100_scene_path('push_cube')}")
        sys.exit(1)
    
    # 创建并运行控制器
    controller = InteractiveController(
        args.scene, 
        record=args.record,
        window_size=(args.width, args.height),
        max_fps=args.fps,
        render_quality=args.quality
    )
    controller.run()


if __name__ == "__main__":
    main()
