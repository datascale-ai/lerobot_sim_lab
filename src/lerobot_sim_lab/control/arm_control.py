#!/usr/bin/env python3
"""
MuJoCo SO-100 arm control CLI (installed as ``lerobot-sim-control``).

Supports ``--mode keyboard`` (MuJoCo viewer + keys), ``--mode remote`` (TCP
joint deltas from a client), and ``--mode send`` (stream to a real SO-100 via
LeRobot, optional LeRobot-format recording).
"""

import argparse
import json
import select
import socket
import time
from pathlib import Path

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

SCENE_XML = str(get_so100_models_dir() / "scene.xml")

JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
SEND_RECORD_CAMERAS = ("camera_front_new", "camera_side")
SEND_RECORD_WIDTH = 640
SEND_RECORD_HEIGHT = 480

CONTROL_STEP = 0.05
GRIPPER_STEP = 0.1

def set_pens_positions(model, data, pens_config):
    data.qpos[BOX_QPOS_START:BOX_QPOS_START+3] = BOX_POSITION
    data.qpos[BOX_QPOS_START+3:BOX_QPOS_START+7] = BOX_QUATERNION
    
    for pen_name, (pos, quat) in pens_config.items():
        if pen_name in PEN_QPOS_MAP:
            qpos_start = PEN_QPOS_MAP[pen_name]
            data.qpos[qpos_start:qpos_start+3] = pos
            data.qpos[qpos_start+3:qpos_start+7] = quat
    
    mujoco.mj_forward(model, data)


class KeyboardController:
    def __init__(self, scenario_id: int):
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data = mujoco.MjData(self.model)
        
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        
        self.joint_ids = [self.model.joint(name).id for name in JOINT_NAMES]
        self.actuator_ids = [self.model.actuator(name).id for name in JOINT_NAMES]
        
        self.current_joint = 0
        
        scenario = next(s for s in PEN_SCENARIOS if s['id'] == scenario_id)
        set_pens_positions(self.model, self.data, scenario['pens'])
        
        print("\n" + "="*60)
        print("MuJoCo SO100 Robot Arm Keyboard Control")
        print("="*60)
        print("\nControls:")
        print("  Letter keys to select joint:")
        print("    Z - Rotation (Base rotation)")
        print("    X - Pitch (Shoulder pitch)")
        print("    C - Elbow")
        print("    V - Wrist_Pitch")
        print("    B - Wrist_Roll")
        print("    N - Jaw (Gripper)")
        print("\n  Arrow keys to control:")
        print("    ← / → : Decrease/Increase current joint angle")
        print("    ↑ / ↓ : Close/Open gripper (only when gripper selected)")
        print("\n  Other:")
        print("    R: Reset to home position")
        print("    Space: Show current status")
        print("    ESC: Exit")
        print("\n  Tip: Number keys 1-6 toggle visibility of geom groups")
        print("="*60 + "\n")
        
    def reset_to_home(self):
        """Reset to home position"""
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        print("Reset to home position")
        
    def show_help(self):
        """Show help information"""
        joint_keys = ['Z', 'X', 'C', 'V', 'B', 'N']
        print("\n" + "="*60)
        print("Current joint status:")
        for i, name in enumerate(JOINT_NAMES):
            qpos_idx = self.model.joint(name).qposadr[0]
            current_val = self.data.qpos[qpos_idx]
            ctrl_val = self.data.ctrl[i]
            joint_range = self.model.jnt_range[self.joint_ids[i]]
            selected = ">>> " if i == self.current_joint else "    "
            print(f"{selected}[{joint_keys[i]}] {name:15s}: qpos={current_val:7.3f}, "
                  f"ctrl={ctrl_val:7.3f}, range=[{joint_range[0]:6.2f}, {joint_range[1]:6.2f}]")
        print(f"\nCurrently selected: [{joint_keys[self.current_joint]}] {JOINT_NAMES[self.current_joint]}")
        print("="*60 + "\n")
    
    def update_control(self, joint_idx, delta):
        """Update control value for specified joint"""
        joint_keys = ['Z', 'X', 'C', 'V', 'B', 'N']
        joint_name = JOINT_NAMES[joint_idx]
        joint_range = self.model.jnt_range[self.joint_ids[joint_idx]]
        
        new_ctrl = self.data.ctrl[joint_idx] + delta
        new_ctrl = np.clip(new_ctrl, joint_range[0], joint_range[1])
        
        self.data.ctrl[joint_idx] = new_ctrl
        
        print(f"[{joint_keys[joint_idx]}] {joint_name}: {new_ctrl:.3f} (range: [{joint_range[0]:.2f}, {joint_range[1]:.2f}])")
    
    def key_callback(self, keycode):
        """Keyboard callback function"""
        joint_key_map = {
            ord('Z'): 0, ord('z'): 0,
            ord('X'): 1, ord('x'): 1,
            ord('C'): 2, ord('c'): 2,
            ord('V'): 3, ord('v'): 3,
            ord('B'): 4, ord('b'): 4,
            ord('N'): 5, ord('n'): 5,
        }
        
        if keycode == 256:  # ESC
            return False
        elif keycode == ord('R') or keycode == ord('r'):
            self.reset_to_home()
        elif keycode == 32:  # Space bar
            self.show_help()
        elif keycode in joint_key_map:
            self.current_joint = joint_key_map[keycode]
            joint_keys = ['Z', 'X', 'C', 'V', 'B', 'N']
            print(f"\nSelected joint [{joint_keys[self.current_joint]}]: {JOINT_NAMES[self.current_joint]}")
        elif keycode == 263:  # Left arrow
            self.update_control(self.current_joint, -CONTROL_STEP)
        elif keycode == 262:  # Right arrow
            self.update_control(self.current_joint, CONTROL_STEP)
        elif keycode == 265:  # Up arrow (close gripper)
            if self.current_joint == 5:
                self.update_control(5, -GRIPPER_STEP)
            else:
                print("Up/Down arrows only work when gripper is selected")
        elif keycode == 264:  # Down arrow (open gripper)
            if self.current_joint == 5:
                self.update_control(5, GRIPPER_STEP)
            else:
                print("Up/Down arrows only work when gripper is selected")
        
        return True
    
    def run(self):
        """Run the controller"""
        with mujoco.viewer.launch_passive(self.model, self.data, 
                                          key_callback=self.key_callback) as viewer:
            viewer.cam.azimuth = 140
            viewer.cam.elevation = -20
            viewer.cam.distance = 1.5
            viewer.cam.lookat[:] = [0.0, -0.3, 0.8]
            
            self.show_help()
            
            while viewer.is_running():
                step_start = time.time()
                
                mujoco.mj_step(self.model, self.data)
                
                viewer.sync()
                
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


class RemoteController:
    def __init__(self, scenario_id: int, host: str, port: int):
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.joint_ids = [self.model.joint(name).id for name in JOINT_NAMES]
        self.joint_ranges = [self.model.jnt_range[jid] for jid in self.joint_ids]

        scenario = next(s for s in PEN_SCENARIOS if s['id'] == scenario_id)
        set_pens_positions(self.model, self.data, scenario['pens'])

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
        self.server.listen(1)
        self.server.setblocking(False)
        self.conn = None
        self.buffer = ""
        self.host = host
        self.port = port

    def _apply_ctrl(self, ctrl):
        if len(ctrl) != len(JOINT_NAMES):
            return
        for i, value in enumerate(ctrl):
            r = self.joint_ranges[i]
            self.data.ctrl[i] = np.clip(float(value), r[0], r[1])

    def _poll_socket(self):
        if self.conn is None:
            try:
                conn, addr = self.server.accept()
                conn.setblocking(False)
                self.conn = conn
                print(f"Client connected: {addr[0]}:{addr[1]}")
            except BlockingIOError:
                return
        if self.conn is None:
            return
        try:
            if not select.select([self.conn], [], [], 0)[0]:
                return
            data = self.conn.recv(4096)
            if not data:
                self.conn.close()
                self.conn = None
                self.buffer = ""
                print("Client disconnected")
                return
            self.buffer += data.decode('utf-8', errors='ignore')
            while '\n' in self.buffer:
                line, self.buffer = self.buffer.split('\n', 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, list):
                    self._apply_ctrl(payload)
        except OSError:
            if self.conn is not None:
                self.conn.close()
            self.conn = None
            self.buffer = ""

    def run(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.azimuth = 140
            viewer.cam.elevation = -20
            viewer.cam.distance = 1.5
            viewer.cam.lookat[:] = [0.0, -0.3, 0.8]
            print(f"Listening on {self.host}:{self.port}")
            while viewer.is_running():
                step_start = time.time()
                self._poll_socket()
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


class SendModeRecorder:
    def __init__(self, scenario_id: int, fps: int, duration_s: float, output_path: str | None):
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        scenario = next(s for s in PEN_SCENARIOS if s['id'] == scenario_id)
        set_pens_positions(self.model, self.data, scenario['pens'])
        self.cameras = SEND_RECORD_CAMERAS
        for camera_name in self.cameras:
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name) < 0:
                raise ValueError(f"camera not found: {camera_name}")
        self.renderers = [
            mujoco.Renderer(self.model, height=SEND_RECORD_HEIGHT, width=SEND_RECORD_WIDTH)
            for _ in self.cameras
        ]
        self.video_fps = fps if fps > 0 else 30
        self.max_frames = max(1, int(round(duration_s * self.video_fps)))
        self.n_substeps = max(1, int(round((1.0 / self.video_fps) / self.model.opt.timestep)))
        self.frames = []
        if output_path:
            self.output_path = Path(output_path).expanduser()
        else:
            self.output_path = Path.cwd() / f"send_dual_view_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        if self.output_path.suffix.lower() != ".mp4":
            self.output_path = self.output_path.with_suffix(".mp4")
        self.saved = False

    def capture(self, ctrl):
        if self.saved:
            return
        self.data.ctrl[:len(ctrl)] = ctrl
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        frames = []
        for renderer, camera_name in zip(self.renderers, self.cameras):
            renderer.update_scene(self.data, camera=camera_name)
            frames.append(renderer.render())
        self.frames.append(np.hstack(frames))
        if len(self.frames) >= self.max_frames:
            self.save()

    def save(self):
        if self.saved or not self.frames:
            return
        import cv2

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        height, width = self.frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(self.output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.video_fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"failed to open video writer: {self.output_path}")
        for frame in self.frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        self.saved = True
        print(f"Saved dual-view video: {self.output_path}")

    def close(self):
        self.save()
        for renderer in self.renderers:
            renderer.close()


def send_real_robot_stream(
    scenario_id: int,
    server_host: str,
    server_port: int,
    robot_port: str,
    robot_id: str,
    fps: int,
    robot_type: str,
    record_seconds: float,
    record_output: str | None,
):
    if robot_type == "so100_leader":
        from lerobot.teleoperators.so100_leader.config_so100_leader import SO100LeaderConfig
        from lerobot.teleoperators.so100_leader.so100_leader import SO100Leader
        config = SO100LeaderConfig(port=robot_port, id=robot_id)
        robot = SO100Leader(config)
        read_fn = robot.get_action
    else:
        from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
        from lerobot.robots.so100_follower.so100_follower import SO100Follower
        config = SO100FollowerConfig(port=robot_port, id=robot_id)
        robot = SO100Follower(config)
        read_fn = robot.get_observation

    robot.connect()
    motor_keys = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((server_host, server_port))

    target_model = mujoco.MjModel.from_xml_path(SCENE_XML)
    joint_ids = [target_model.joint(name).id for name in JOINT_NAMES]
    joint_ranges = [target_model.jnt_range[jid] for jid in joint_ids]
    recorder = None
    if record_seconds > 0:
        recorder = SendModeRecorder(scenario_id, fps, record_seconds, record_output)
        print(
            f"Recording {record_seconds:.1f}s dual-view video from "
            f"{SEND_RECORD_CAMERAS[0]} + {SEND_RECORD_CAMERAS[1]}"
        )

    try:
        while True:
            t0 = time.perf_counter()
            obs = read_fn()
            values = [obs.get(f"{k}.pos") for k in motor_keys]
            if any(v is None for v in values):
                continue
            ctrl = []
            for i, v in enumerate(values):
                r = joint_ranges[i]
                if i == 5:
                    mapped = r[0] + (float(v) / 100.0) * (r[1] - r[0])
                else:
                    mapped = r[0] + ((float(v) + 100.0) / 200.0) * (r[1] - r[0])
                ctrl.append(mapped)
            sock.sendall((json.dumps(ctrl) + "\n").encode("utf-8"))
            if recorder is not None:
                recorder.capture(ctrl)
            dt = time.perf_counter() - t0
            if fps > 0:
                time.sleep(max(0.0, 1.0 / fps - dt))
    finally:
        if recorder is not None:
            recorder.close()
        sock.close()
        if hasattr(robot, "disconnect"):
            robot.disconnect()


def main():
    parser = argparse.ArgumentParser(
        description="MuJoCo SO-100 arm control (keyboard, TCP remote, or send to real robot).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  lerobot-sim-control --scenario 0
  lerobot-sim-control --scenario 0 --mode remote --listen-port 5555
  lerobot-sim-control --scenario 0 --mode send --robot-port /dev/ttyUSB0 \\
      --server-host 127.0.0.1 --server-port 5555
""".strip(),
    )
    parser.add_argument('--scenario', type=int, required=True, choices=[0,1,2,3,4,5])
    parser.add_argument('--mode', type=str, default='keyboard', choices=['keyboard', 'remote', 'send'])
    parser.add_argument('--listen-host', type=str, default='0.0.0.0')
    parser.add_argument('--listen-port', type=int, default=5555)
    parser.add_argument('--server-host', type=str, default='127.0.0.1')
    parser.add_argument('--server-port', type=int, default=5555)
    parser.add_argument('--robot-port', type=str, default='')
    parser.add_argument('--robot-id', type=str, default='so100')
    parser.add_argument('--robot-type', type=str, default='so100_leader', choices=['so100_leader', 'so100_follower'])
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--record-seconds', type=float, default=0.0)
    parser.add_argument('--record-output', type=str, default='')
    args = parser.parse_args()
    if args.mode == 'keyboard':
        controller = KeyboardController(args.scenario)
        controller.run()
    elif args.mode == 'remote':
        controller = RemoteController(args.scenario, args.listen_host, args.listen_port)
        controller.run()
    else:
        if not args.robot_port:
            raise SystemExit("robot-port required for send mode")
        send_real_robot_stream(
            args.scenario,
            args.server_host,
            args.server_port,
            args.robot_port,
            args.robot_id,
            args.fps,
            args.robot_type,
            args.record_seconds,
            args.record_output or None,
        )


if __name__ == "__main__":
    main()
