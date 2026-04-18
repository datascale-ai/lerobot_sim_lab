import gymnasium as gym
import mujoco
import numpy as np
from PIL import Image


def test_camera_positions():
    env = gym.make("lerobot_sim_lab/SO100PickCube-v0", image_obs=True, render_mode="rgb_array")
    obs, _ = env.reset(seed=42)
    model = env.unwrapped._model
    renderer = mujoco.Renderer(model, height=480, width=640)
    
    positions = {
        "x_plus":  (90, "Right (+X)"),
        "x_minus": (270, "Left (-X)"),
        "y_plus":  (180, "Front (+Y)"),
        "y_minus": (0, "Back (-Y)"),
    }
    
    for name, (az, desc) in positions.items():
        print(f"渲染位置 {name} (Az {az}, {desc})...")
        
        cam = mujoco.MjvCamera()
        # 不设置 cam.type，或者设为 mjCAMERA_FREE
        # 只要设置了 lookat, distance, azimuth, elevation，它就是自由相机
        
        cam.lookat = np.array([0, 0.08, 0.15]) # 保持和录制脚本一致的 lookat
        cam.distance = 0.6
        cam.elevation = -10
        cam.azimuth = az
        
        renderer.update_scene(env.unwrapped._data, camera=cam)
        img = renderer.render()
        Image.fromarray(img).save(f"debug_cam_{name}.png")

    env.close()

if __name__ == "__main__":
    test_camera_positions()
