"""
MuJoCo 仿真接收器 - WebSocket 版本
提供更现代的 WebSocket 接口，支持双向通信
"""

import asyncio
import json
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import websockets


class WebSocketSimReceiver:
    def __init__(self, model_path="models/robot_arm.xml", port=8765):
        self.port = port
        self.clients = set()
        
        # 加载模型
        if Path(model_path).exists():
            print(f"📂 加载模型: {model_path}")
            self.model = mujoco.MjModel.from_xml_path(model_path)
        else:
            print("⚠️  使用默认测试模型")
            xml = """
            <mujoco>
              <worldbody>
                <body name="link1" pos="0 0 0">
                  <geom type="capsule" size="0.05" fromto="0 0 0 0 0 0.3"/>
                  <joint name="joint1" type="hinge" axis="0 0 1"/>
                  <body name="link2" pos="0 0 0.3">
                    <geom type="capsule" size="0.04" fromto="0 0 0 0.3 0 0"/>
                    <joint name="joint2" type="hinge" axis="0 1 0"/>
                  </body>
                </body>
              </worldbody>
            </mujoco>
            """
            self.model = mujoco.MjModel.from_xml_string(xml)
        
        self.data = mujoco.MjData(self.model)
        self.latest_data = None
        self.packet_count = 0
        self.running = True
        
    async def handle_client(self, websocket, path):
        """处理 WebSocket 客户端连接"""
        self.clients.add(websocket)
        client_addr = websocket.remote_address
        print(f"✅ 客户端已连接: {client_addr}")
        
        try:
            async for message in websocket:
                # 接收机械臂数据
                try:
                    arm_data = json.loads(message)
                    self.latest_data = arm_data
                    self.packet_count += 1
                    
                    # 可选：发送仿真状态反馈
                    response = {
                        'status': 'ok',
                        'packet_id': self.packet_count,
                        'sim_time': self.data.time
                    }
                    await websocket.send(json.dumps(response))
                    
                except json.JSONDecodeError as e:
                    await websocket.send(json.dumps({'status': 'error', 'message': str(e)}))
                    
        except websockets.exceptions.ConnectionClosed:
            print(f"❌ 客户端断开: {client_addr}")
        finally:
            self.clients.remove(websocket)
    
    def update_simulation(self):
        """在独立线程中更新仿真"""
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            last_update = time.time()
            
            while viewer.is_running() and self.running:
                if self.latest_data:
                    # 同步数据到仿真
                    if 'qpos' in self.latest_data:
                        qpos = np.array(self.latest_data['qpos'])
                        n = min(len(qpos), self.model.nq)
                        self.data.qpos[:n] = qpos[:n]
                    
                    if 'qvel' in self.latest_data:
                        qvel = np.array(self.latest_data['qvel'])
                        n = min(len(qvel), self.model.nv)
                        self.data.qvel[:n] = qvel[:n]
                
                # 步进仿真
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                
                # 统计信息
                if time.time() - last_update > 1.0:
                    print(f"📊 更新频率: {self.packet_count} 包/秒 | 客户端: {len(self.clients)}")
                    self.packet_count = 0
                    last_update = time.time()
                
                time.sleep(0.01)  # 100Hz
    
    async def start_server(self):
        """启动 WebSocket 服务器"""
        print("🚀 WebSocket 服务器启动")
        print(f"📡 监听端口: {self.port}")
        print(f"🌐 连接地址: ws://localhost:{self.port}")
        
        # 在独立线程中运行仿真
        sim_thread = threading.Thread(target=self.update_simulation, daemon=True)
        sim_thread.start()
        
        # 启动 WebSocket 服务器
        async with websockets.serve(self.handle_client, "0.0.0.0", self.port):
            await asyncio.Future()  # 永久运行


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="WebSocket 仿真接收器")
    parser.add_argument('--model', type=str, default='models/robot_arm.xml')
    parser.add_argument('--port', type=int, default=8765)
    
    args = parser.parse_args()
    
    receiver = WebSocketSimReceiver(model_path=args.model, port=args.port)
    
    try:
        asyncio.run(receiver.start_server())
    except KeyboardInterrupt:
        print("\n🛑 服务器已停止")
        receiver.running = False


















