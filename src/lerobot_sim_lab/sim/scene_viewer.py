#!/usr/bin/env python3
"""
快速查看 MuJoCo 场景文件
通过 VNC 窗口可视化场景

用法:
    python -m lerobot_sim_lab.sim.scene_viewer
    python -m lerobot_sim_lab.sim.scene_viewer --xml push_cube_loop.xml
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer

from lerobot_sim_lab.utils.paths import get_so100_models_dir


def print_scene_info(model: mujoco.MjModel):
    """打印场景信息"""
    print("\n" + "=" * 80)
    print("场景信息分析")
    print("=" * 80)

    # 基本信息
    print("\n📊 基本统计:")
    print(f"  - 总自由度 (nq): {model.nq}")
    print(f"  - 总速度维度 (nv): {model.nv}")
    print(f"  - 控制器数量 (nu): {model.nu}")
    print(f"  - 时间步长 (timestep): {model.opt.timestep}s")

    # Bodies (刚体)
    print(f"\n🎯 Bodies (共 {model.nbody} 个):")
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        print(f"  [{i}] {body_name}")

    # Joints (关节)
    print(f"\n🔗 Joints (共 {model.njnt} 个):")
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        joint_type = model.jnt_type[i]
        type_names = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
        print(f"  [{i}] {joint_name} - {type_names.get(joint_type, 'unknown')}")

    # Geoms (几何体)
    print(f"\n📐 Geoms (共 {model.ngeom} 个):")
    for i in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        geom_type = model.geom_type[i]
        type_names = {0: "plane", 1: "hgeom", 2: "sphere", 3: "capsule", 5: "cylinder", 6: "box", 7: "mesh"}
        print(f"  [{i}] {geom_name} - {type_names.get(geom_type, 'unknown')}")

    # Cameras (相机)
    if model.ncam > 0:
        print(f"\n📷 Cameras (共 {model.ncam} 个):")
        for i in range(model.ncam):
            cam_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            print(f"  [{i}] {cam_name}")

    # Actuators (执行器)
    if model.nu > 0:
        print(f"\n⚙️  Actuators (共 {model.nu} 个):")
        for i in range(model.nu):
            act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            print(f"  [{i}] {act_name}")

    print("\n" + "=" * 80)


def visualize_scene(xml_path: str):
    """可视化 MuJoCo 场景"""
    xml_file = Path(xml_path)
    if not xml_file.is_absolute() and not xml_file.exists():
        xml_file = get_so100_models_dir() / xml_path

    if not xml_file.exists():
        print(f"❌ 错误: 文件不存在: {xml_path}")
        return

    print(f"📂 加载场景: {xml_file.name}")
    print(f"📍 完整路径: {xml_file.absolute()}")

    try:
        # 加载模型
        model = mujoco.MjModel.from_xml_path(str(xml_file))
        data = mujoco.MjData(model)

        # 打印场景信息
        print_scene_info(model)

        # 重置到初始状态
        mujoco.mj_resetData(model, data)

        # 如果有keyframe，加载home姿态
        if model.nkey > 0:
            key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
            if key_id >= 0:
                mujoco.mj_resetDataKeyframe(model, data, key_id)
                print("\n✅ 已加载 keyframe: home")

        print("\n🎮 控制说明:")
        print("  - 鼠标左键拖动: 旋转视角")
        print("  - 鼠标右键拖动: 平移视角")
        print("  - 鼠标滚轮: 缩放")
        print("  - 双击物体: 聚焦到该物体")
        print("  - Space: 暂停/继续仿真")
        print("  - Backspace: 重置场景")
        print("  - Ctrl+Q 或关闭窗口: 退出")
        print("\n🚀 正在启动交互式查看器...")
        print("=" * 80 + "\n")

        # 启动交互式查看器
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # 设置相机视角（俯视角度）
            viewer.cam.azimuth = 120
            viewer.cam.elevation = -20
            viewer.cam.distance = 2.5
            viewer.cam.lookat[:] = [0, 0.2, 0.5]

            # 主循环
            while viewer.is_running():
                step_start = time.time()

                # 物理仿真步进
                mujoco.mj_step(model, data)

                # 同步查看器
                viewer.sync()

                # 控制仿真速度（实时）
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

        print("\n👋 查看器已关闭")

    except Exception as e:
        print(f"\n❌ 加载场景失败: {e}")
        import traceback

        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="MuJoCo 场景可视化工具")
    parser.add_argument(
        "--xml", type=str, default="scene.xml", help="XML场景文件名或绝对路径"
    )
    args = parser.parse_args()

    visualize_scene(args.xml)


if __name__ == "__main__":
    main()
