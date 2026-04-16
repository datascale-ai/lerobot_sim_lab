# Remote Arm Control Quickstart

1. Install the package in your environment:

```bash
pip install -e ".[remote]"
```

2. Start the simulator-side receiver:

```bash
python -m lerobot_sim_lab.control.remote.sim_receiver_socket
```

3. Start the sender on the controller side:

```bash
python -m lerobot_sim_lab.control.remote.local_arm_sender_socket
```

4. For websocket-based transport, replace both commands with the corresponding `*_websocket` modules.

5. Copy and adapt the sample config if needed:

```bash
cp assets/configs/remote_arm_control.config.yaml.example remote_arm_control.config.yaml
```
