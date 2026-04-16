
import threading
import time

import mujoco.viewer
import numpy as np

from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
from gym_lowcostrobot.interface.simulated_robot import SimulatedRobot
from lerobot_sim_lab.utils.paths import get_so100_scene_path


def read_so100_arm_position():
    global target_pos
    while True:
        target_pos = np.array(so100_arm.read("Present_Position"))
        target_pos = (target_pos / 2048 - 1) * 3.14
        # initial post [0 -3.14 3.14 0.817 0 -0.157]
        # Adjust the plus-minus sign and the offsets
        # target_pos[i] = ± target_pos[i] + 0
        
        target_pos[0] = -target_pos[0] + 1.77
        target_pos[1] =  target_pos[1] - 0.45
        target_pos[2] =  target_pos[2] + 1.91
        target_pos[3] =  target_pos[3] + 0.15
        target_pos[4] =  target_pos[4] + 0.622
        target_pos[5] =  target_pos[5] - 0.626
        

if __name__ == "__main__":

    config = FeetechMotorsBusConfig(
        port='/dev/ttyACM0',
        motors={
            # name: (index, model)
            "shoulder_pan": [1, "sts3215"],
            "shoulder_lift": [2, "sts3215"],
            "elbow_flex": [3, "sts3215"],
            "wrist_flex": [4, "sts3215"],
            "wrist_roll": [5, "sts3215"],
            "gripper": [6, "sts3215"],
        },
    )
    so100_arm = FeetechMotorsBus(config)


    # Open port
    if not so100_arm.is_connected:
        so100_arm.connect()

   # Create a MuJoCo model and data
    m = mujoco.MjModel.from_xml_path(str(get_so100_scene_path("push_cube")))
    d = mujoco.MjData(m)
    r = SimulatedRobot(m, d)

    target_pos = np.zeros(6)

    # Start the thread for reading so100_arm position
    so100_arm_thread = threading.Thread(target=read_so100_arm_position)
    so100_arm_thread.daemon = True
    so100_arm_thread.start()

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running():
            # Use the latest target_pos
            step_start = time.time()
            target_pos_local = target_pos.copy()
            # print(f'target pos copy {time.time() - step_start}')

            r.set_target_qpos(target_pos_local)
            # print(f'set targtee pos copy {time.time() - step_start}')

            mujoco.mj_step(m, d)
            # print(f'mjstep {time.time() - step_start}')

            viewer.sync()
            # print(f'viewer sync {time.time() - step_start}')

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            # print(f'time until next step {time_until_next_step}')
            
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    # Close port
    so100_arm.disconnect()
