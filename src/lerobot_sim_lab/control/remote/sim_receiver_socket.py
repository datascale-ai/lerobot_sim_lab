"""
MuJoCo 仿真接收器 - Socket 版本
在 Docker 容器内运行，接收本地机械臂数据并同步到仿真环境
"""

import json
import socket
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


class SimulationReceiver:
    def __init__(self, model_path="models/robot_arm.xml", port=9999):
        """
        初始化仿真接收器
        
        Args:
            model_path: MuJoCo 模型文件路径
            port: 监听端口
        """
        self.port = port
        
        # 加载 MuJoCo 模型
        if Path(model_path).exists():
            print(f"📂 加载模型: {model_path}")
            self.model = mujoco.MjModel.from_xml_path(model_path)
        else:
            print("⚠️  模型文件不存在，使用默认测试模型")
            # 创建一个简单的测试模型
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
        
        # 统计信息
        self.packet_count = 0
        self.start_time = time.time()
        
    def start(self):
        """启动服务器并开始接收数据"""
        # 创建 Socket 服务器
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('0.0.0.0', self.port))
        server.listen(1)
        
        print("🚀 仿真接收器已启动")
        print(f"📡 监听端口: {self.port}")
        print(f"🤖 关节数量: {self.model.nq}")
        print("⏳ 等待本地机械臂连接...")
        
        conn, addr = server.accept()
        print(f"✅ 已连接: {addr}")
        
        # 启动 MuJoCo 可视化
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            buffer = ""
            last_update = time.time()
            
            while viewer.is_running():
                try:
                    # 接收数据
                    chunk = conn.recv(4096).decode('utf-8')
                    if not chunk:
                        print("❌ 连接断开")
                        break
                    
                    buffer += chunk
                    
                    # 处理完整的 JSON 消息（按行分割）
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        try:
                            arm_data = json.loads(line)
                            
                            # 同步关节位置
                            if 'qpos' in arm_data:
                                qpos = np.array(arm_data['qpos'])
                                n = min(len(qpos), self.model.nq)
                                self.data.qpos[:n] = qpos[:n]
                            
                            # 同步关节速度（可选）
                            if 'qvel' in arm_data:
                                qvel = np.array(arm_data['qvel'])
                                n = min(len(qvel), self.model.nv)
                                self.data.qvel[:n] = qvel[:n]
                            
                            # 步进仿真
                            mujoco.mj_step(self.model, self.data)
                            viewer.sync()
                            
                            # 统计信息
                            self.packet_count += 1
                            if time.time() - last_update > 1.0:
                                elapsed = time.time() - self.start_time
                                fps = self.packet_count / elapsed
                                print(f"📊 接收频率: {fps:.1f} Hz | 总包数: {self.packet_count}")
                                last_update = time.time()
                            
                        except json.JSONDecodeError as e:
                            print(f"⚠️  JSON 解析错误: {e}")
                            
                except Exception as e:
                    print(f"❌ 错误: {e}")
                    break
        
        conn.close()
        server.close()
        print("🛑 服务器已关闭")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MuJoCo 仿真接收器")
    parser.add_argument('--model', type=str, default='models/robot_arm.xml',
                        help='MuJoCo 模型文件路径')
    parser.add_argument('--port', type=int, default=9999,
                        help='监听端口')
    
    args = parser.parse_args()
    
    receiver = SimulationReceiver(model_path=args.model, port=args.port)
    receiver.start()


















