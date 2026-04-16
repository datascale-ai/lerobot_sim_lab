import mujoco
import numpy as np
import mediapy as media

model = mujoco.MjModel.from_xml_path("./so101_new_calib.xml")
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)

renderer = mujoco.Renderer(model, height=480, width=640)
frames = []

duration = 5.0
framerate = 30
total_frames = int(duration * framerate)

for i in range(total_frames):
    mujoco.mj_step(model, data)
    if i % 2 == 0:
        renderer.update_scene(data)
        frames.append(renderer.render())

renderer.close()
media.write_video("./scene_output.mp4", frames, fps=framerate)
print(f"✅ 视频已保存: ./scene_output.mp4 ({len(frames)} 帧)")

