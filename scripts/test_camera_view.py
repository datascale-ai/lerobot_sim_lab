import gymnasium as gym
import numpy as np
import mujoco
from PIL import Image
import os

from lerobot_sim_lab.data.record_episodes import render_dual_view

def test_views():
    # 创建环境
    # 注意：这里我们不需要 TimeLimit，直接用 raw env 以方便访问 model
    env = gym.make("lerobot_sim_lab/SO100PickCube-v0", image_obs=True, render_mode="rgb_array")
    obs, _ = env.reset(seed=42)
    
    # 1. 使用动态相机渲染 (Ground Truth)
    print("渲染动态相机视图 (Target)...")
    model = env.unwrapped._model
    renderer = mujoco.Renderer(model, height=480, width=640)
    
    gt_front, gt_side = render_dual_view(env, renderer)
    
    Image.fromarray(gt_front).save("view_gt_front.png")
    Image.fromarray(gt_side).save("view_gt_side.png")
    print("已保存 view_gt_front.png, view_gt_side.png")
    
    # 2. 使用 XML 相机渲染 (Candidate)
    print("渲染 XML 相机视图 (Candidate)...")
    # 我们需要手动指定相机ID进行渲染
    
    def render_xml_camera(camera_name):
        try:
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            renderer.update_scene(env.unwrapped._data, camera=cam_id)
            return renderer.render()
        except Exception as e:
            print(f"无法渲染相机 {camera_name}: {e}")
            return np.zeros((480, 640, 3), dtype=np.uint8)

    xml_front = render_xml_camera("camera_front_new")
    xml_side = render_xml_camera("camera_side")
    
    Image.fromarray(xml_front).save("view_xml_front.png")
    Image.fromarray(xml_side).save("view_xml_side.png")
    print("已保存 view_xml_front.png, view_xml_side.png")
    
    env.close()

if __name__ == "__main__":
    test_views()

