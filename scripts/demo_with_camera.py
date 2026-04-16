"""
SO-100 演示 - 支持多种场景和摄像机角度
"""
import sys

import numpy as np
import mujoco
import mediapy as media
from IPython.display import Video, display
import matplotlib.pyplot as plt

from lerobot_sim_lab.utils.paths import get_so100_scene_path, resolve_output_path


class SO100SimulatorWithCamera:
    """带摄像机控制的 SO-100 仿真器"""
    
    def __init__(self, scene="initial", camera="default"):
        """
        初始化仿真器
        
        Args:
            scene: 场景选择 ("basic", "initial", "push_cube")
            camera: 摄像机视角
                - "default": 默认视角
                - "front": 正面视角
                - "top": 俯视图
                - "side": 侧面视角
                - "close": 近距离视角
                - "wide": 广角视角
                - (pos, lookat): 自定义位置和目标点
        """
        # 场景文件
        xml_path = get_so100_scene_path(scene if scene in {"basic", "initial", "push_cube"} else "initial")
        print(f"加载场景: {xml_path}")
        
        # 加载模型
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # 重置到关键帧
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        
        # 创建渲染器
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        
        # 设置摄像机
        self.set_camera(camera)
        
        print(f"✅ 仿真器初始化完成")
        print(f"   场景: {scene}")
        print(f"   摄像机: {camera}")
        print(f"   关节数: {self.model.njnt}")
        print(f"   自由度: {self.model.nq}")
    
    def set_camera(self, camera):
        """设置摄像机位置"""
        if isinstance(camera, tuple) and len(camera) == 2:
            # 自定义摄像机
            pos, lookat = camera
            self.renderer.update_scene(self.data, camera=mujoco.MjvCamera())
            self.renderer.scene.camera[0].lookat = lookat
            self.renderer.scene.camera[0].distance = np.linalg.norm(np.array(pos) - np.array(lookat))
            self.renderer.scene.camera[0].azimuth = 0
            self.renderer.scene.camera[0].elevation = -20
        elif camera == "front":
            # 正面视角
            self.camera_config = {
                "lookat": np.array([0, 0, 0.15]),
                "distance": 0.6,
                "azimuth": 90,
                "elevation": -10
            }
        elif camera == "top":
            # 俯视图
            self.camera_config = {
                "lookat": np.array([0, 0, 0.15]),
                "distance": 0.8,
                "azimuth": 90,
                "elevation": -80
            }
        elif camera == "side":
            # 侧面视角
            self.camera_config = {
                "lookat": np.array([0, 0, 0.15]),
                "distance": 0.6,
                "azimuth": 0,
                "elevation": -10
            }
        elif camera == "close":
            # 近距离特写
            self.camera_config = {
                "lookat": np.array([0, 0, 0.2]),
                "distance": 0.3,
                "azimuth": 120,
                "elevation": -15
            }
        elif camera == "wide":
            # 广角全景
            self.camera_config = {
                "lookat": np.array([0, 0, 0.1]),
                "distance": 1.2,
                "azimuth": 135,
                "elevation": -30
            }
        else:
            # 默认视角
            self.camera_config = {
                "lookat": np.array([0, 0, 0.15]),
                "distance": 0.7,
                "azimuth": 135,
                "elevation": -20
            }
    
    def render_frame(self):
        """渲染当前帧"""
        # 应用摄像机配置
        if hasattr(self, 'camera_config'):
            cam = mujoco.MjvCamera()
            cam.lookat = self.camera_config["lookat"]
            cam.distance = self.camera_config["distance"]
            cam.azimuth = self.camera_config["azimuth"]
            cam.elevation = self.camera_config["elevation"]
            self.renderer.update_scene(self.data, camera=cam)
        else:
            self.renderer.update_scene(self.data)
        
        return self.renderer.render()
    
    def step(self, n_steps=1):
        """执行仿真步骤"""
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)
    
    def set_control(self, target):
        """设置控制目标"""
        self.data.ctrl[:6] = target
    
    def get_joint_positions(self):
        """获取关节位置"""
        return self.data.qpos[:6].copy()
    
    def show_frame(self):
        """显示当前帧"""
        frame = self.render_frame()
        plt.figure(figsize=(10, 6))
        plt.imshow(frame)
        plt.axis('off')
        plt.title(f'Camera: {self.camera_config if hasattr(self, "camera_config") else "default"}')
        plt.tight_layout()
        plt.show()


def demo_initial_scene():
    """演示 initial 场景"""
    print("=" * 60)
    print("SO-100 Initial 场景演示")
    print("=" * 60)
    
    # 创建仿真器 - initial 场景，默认视角
    sim = SO100SimulatorWithCamera(scene="initial", camera="default")
    
    # 显示初始状态
    print("\n初始关节位置:")
    print(sim.get_joint_positions())
    
    print("\n显示初始帧...")
    sim.show_frame()
    
    # 定义一些动作
    print("\n执行动作序列...")
    
    actions = [
        (np.array([0, -3.14, 3.14, 0.817, 0, -0.157]), 50, "初始姿态"),
        (np.array([0.8, -2.5, 2.8, 0.6, 0.5, -0.157]), 100, "右侧摆动"),
        (np.array([-0.8, -2.5, 2.8, 0.6, -0.5, -0.157]), 100, "左侧摆动"),
        (np.array([0, -2.0, 2.0, 0.3, 0, -0.157]), 100, "前倾"),
        (np.array([0, -3.0, 3.0, 1.0, 0, 0.5]), 100, "后仰+张开夹爪"),
        (np.array([0, -3.14, 3.14, 0.817, 0, -0.157]), 100, "回到初始"),
    ]
    
    frames = []
    joint_history = []
    
    for i, (target, steps, desc) in enumerate(actions):
        print(f"  阶段 {i+1}/{len(actions)}: {desc}")
        sim.set_control(target)
        
        for step in range(steps):
            sim.step()
            joint_history.append(sim.get_joint_positions())
            
            if step % 2 == 0:
                frames.append(sim.render_frame())
    
    joint_history = np.array(joint_history)
    print(f"\n✅ 完成，共 {len(frames)} 帧")
    
    # 保存视频
    video_path = str(resolve_output_path("initial_scene_default.mp4"))
    media.write_video(video_path, frames, fps=30)
    print(f"视频保存: {video_path}")
    
    return Video(video_path, embed=True, width=640)


def demo_camera_angles():
    """演示不同摄像机角度"""
    print("=" * 60)
    print("多角度摄像机演示")
    print("=" * 60)
    
    # 定义摄像机视角
    cameras = {
        "default": "默认视角",
        "front": "正面视角",
        "top": "俯视图",
        "side": "侧面视角",
        "close": "近距离",
        "wide": "广角全景"
    }
    
    # 创建一个动作序列（通用）
    action_sequence = [
        (np.array([0, -3.14, 3.14, 0.817, 0, -0.157]), 30),
        (np.array([0.8, -2.5, 2.8, 0.6, 0, -0.157]), 80),
        (np.array([-0.8, -2.5, 2.8, 0.6, 0, -0.157]), 80),
        (np.array([0, -2.5, 2.5, 0.5, 1.5, -0.157]), 80),
        (np.array([0, -3.14, 3.14, 0.817, 0, -0.157]), 60),
    ]
    
    # 为每个视角生成视频
    video_paths = {}
    
    for cam_name, cam_desc in cameras.items():
        print(f"\n生成 {cam_desc} 视频...")
        
        # 创建仿真器
        sim = SO100SimulatorWithCamera(scene="push_cube", camera=cam_name)
        
        frames = []
        
        # 执行动作
        for target, steps in action_sequence:
            sim.set_control(target)
            for _ in range(steps):
                sim.step()
                frames.append(sim.render_frame())
        
        # 保存视频
        video_path = str(resolve_output_path(f"camera_{cam_name}.mp4"))
        media.write_video(video_path, frames, fps=30)
        video_paths[cam_name] = video_path
        print(f"  保存: {video_path}")
    
    print("\n✅ 所有视角视频生成完成！")
    print("\n视频文件:")
    for cam_name, path in video_paths.items():
        print(f"  {cameras[cam_name]}: {path}")
    
    # 显示对比图
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, (cam_name, cam_desc) in enumerate(cameras.items()):
        sim = SO100SimulatorWithCamera(scene="push_cube", camera=cam_name)
        sim.set_control(np.array([0.5, -2.5, 2.8, 0.6, 0, -0.157]))
        for _ in range(100):
            sim.step()
        
        frame = sim.render_frame()
        axes[idx].imshow(frame)
        axes[idx].set_title(cam_desc, fontsize=12, fontweight='bold')
        axes[idx].axis('off')
    
    plt.tight_layout()
    comparison_path = resolve_output_path("camera_comparison.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"\n对比图保存: {comparison_path}")
    plt.show()
    
    return video_paths


def demo_custom_camera():
    """自定义摄像机示例"""
    print("=" * 60)
    print("自定义摄像机角度演示")
    print("=" * 60)
    
    # 自定义摄像机配置
    custom_cameras = [
        {
            "name": "bird_eye",
            "lookat": np.array([0, 0, 0.1]),
            "distance": 1.0,
            "azimuth": 90,
            "elevation": -85,
            "desc": "鸟瞰图"
        },
        {
            "name": "ground_level",
            "lookat": np.array([0, 0, 0.05]),
            "distance": 0.5,
            "azimuth": 90,
            "elevation": 0,
            "desc": "地面视角"
        },
        {
            "name": "diagonal",
            "lookat": np.array([0, 0, 0.2]),
            "distance": 0.8,
            "azimuth": 45,
            "elevation": -45,
            "desc": "对角视角"
        }
    ]
    
    print("\n生成自定义视角...")
    
    for cam_config in custom_cameras:
        print(f"\n{cam_config['desc']} ({cam_config['name']}):")
        print(f"  lookat: {cam_config['lookat']}")
        print(f"  distance: {cam_config['distance']}")
        print(f"  azimuth: {cam_config['azimuth']}, elevation: {cam_config['elevation']}")
        
        # 创建仿真器并手动设置摄像机
        sim = SO100SimulatorWithCamera(scene="push_cube", camera="default")
        sim.camera_config = {
            "lookat": cam_config["lookat"],
            "distance": cam_config["distance"],
            "azimuth": cam_config["azimuth"],
            "elevation": cam_config["elevation"]
        }
        
        # 执行简单动作
        frames = []
        actions = [
            np.array([0, -3.14, 3.14, 0.817, 0, -0.157]),
            np.array([0.8, -2.5, 2.8, 0.6, 0, -0.157]),
            np.array([-0.8, -2.5, 2.8, 0.6, 0, -0.157]),
        ]
        
        for target in actions:
            sim.set_control(target)
            for _ in range(80):
                sim.step()
                frames.append(sim.render_frame())
        
        # 保存
        video_path = str(resolve_output_path(f'custom_{cam_config["name"]}.mp4'))
        media.write_video(video_path, frames, fps=30)
        print(f"  保存: {video_path}")


def show_camera_help():
    """显示摄像机参数说明"""
    help_text = """
    ╔══════════════════════════════════════════════════════════╗
    ║            SO-100 摄像机参数说明                          ║
    ╚══════════════════════════════════════════════════════════╝
    
    预设视角:
    ┌──────────┬────────────────────────────────────────────┐
    │ "default"│ 默认 3/4 视角，观察整体运动                │
    │ "front"  │ 正面视角，观察前后运动                      │
    │ "top"    │ 俯视图，观察平面位置                        │
    │ "side"   │ 侧面视角，观察上下运动                      │
    │ "close"  │ 近距离特写，观察细节                        │
    │ "wide"   │ 广角全景，包含环境                          │
    └──────────┴────────────────────────────────────────────┘
    
    摄像机参数:
    • lookat:    摄像机注视点 [x, y, z]
    • distance:  摄像机距离目标点的距离
    • azimuth:   方位角 (0-360°，水平旋转)
                 0° = 正前方, 90° = 左侧, 180° = 后方, 270° = 右侧
    • elevation: 仰角 (-90 到 90°，垂直旋转)
                 -90° = 俯视, 0° = 水平, 90° = 仰视
    
    使用示例:
    ```python
    # 使用预设视角
    sim = SO100SimulatorWithCamera(scene="initial", camera="top")
    
    # 自定义摄像机
    sim = SO100SimulatorWithCamera(scene="push_cube", camera="default")
    sim.camera_config = {
        "lookat": np.array([0, 0, 0.2]),
        "distance": 0.6,
        "azimuth": 120,
        "elevation": -25
    }
    ```
    """
    print(help_text)


if __name__ == "__main__":
    print("请在 Jupyter Notebook 中运行:")
    print()
    print("from demo_with_camera import demo_initial_scene, demo_camera_angles")
    print("from demo_with_camera import demo_custom_camera, show_camera_help")
    print()
    print("# 查看 initial 场景")
    print("video = demo_initial_scene()")
    print("video")
    print()
    print("# 生成所有预设视角的视频")
    print("videos = demo_camera_angles()")
    print()
    print("# 查看摄像机参数说明")
    print("show_camera_help()")
