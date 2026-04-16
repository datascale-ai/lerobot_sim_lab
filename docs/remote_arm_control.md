# Remote Arm Control

Remote control utilities now live in `src/lerobot_sim_lab/control/remote/`.

## Available Modules

- `local_arm_sender_socket.py`
- `local_arm_sender_websocket.py`
- `sim_receiver_socket.py`
- `sim_receiver_websocket.py`

## Local Usage

From the repository root:

```bash
python -m lerobot_sim_lab.control.remote.sim_receiver_socket
python -m lerobot_sim_lab.control.remote.local_arm_sender_socket
```

Or for the websocket pair:

```bash
python -m lerobot_sim_lab.control.remote.sim_receiver_websocket
python -m lerobot_sim_lab.control.remote.local_arm_sender_websocket
```

## Config

Example config moved to `assets/configs/remote_arm_control.config.yaml.example`.
