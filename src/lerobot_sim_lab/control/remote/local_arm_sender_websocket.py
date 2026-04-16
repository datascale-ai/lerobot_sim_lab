"""
本地机械臂数据发送器 - WebSocket 版本
支持双向通信和更好的错误处理
"""

import argparse
import asyncio
import json
import time

import numpy as np
import websockets

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


class WebSocketArmSender:
    def __init__(self, url, serial_port=None, baudrate=115200):
        self.url = url
        self.serial_conn = None
        self.sim_time = 0.0
        
        if serial_port and SERIAL_AVAILABLE:
            try:
                self.serial_conn = serial.Serial(serial_port, baudrate, timeout=0.1)
                print(f"✅ 串口已连接: {serial_port}")
            except Exception as e:
                print(f"❌ 串口连接失败: {e}")
    
    def read_arm_state(self):
        """读取机械臂状态"""
        self.sim_time += 0.01
        
        # 模拟数据（正弦波）
        qpos = [
            0.5 * np.sin(self.sim_time * 0.5),
            0.3 * np.sin(self.sim_time * 0.8),
            0.4 * np.sin(self.sim_time * 1.0),
            0.2 * np.sin(self.sim_time * 1.2),
            0.3 * np.sin(self.sim_time * 0.9),
            0.5 * np.sin(self.sim_time * 0.7),
        ]
        
        qvel = [0.0] * 6  # 简化
        
        return {
            'qpos': qpos,
            'qvel': qvel,
            'timestamp': time.time()
        }
    
    async def send_loop(self, frequency=100):
        """主发送循环"""
        dt = 1.0 / frequency
        packet_count = 0
        start_time = time.time()
        
        async with websockets.connect(self.url) as websocket:
            print(f"✅ 已连接: {self.url}")
            print(f"🚀 开始发送 @ {frequency} Hz")
            
            try:
                while True:
                    loop_start = time.time()
                    
                    # 读取并发送
                    arm_data = self.read_arm_state()
                    await websocket.send(json.dumps(arm_data))
                    
                    # 接收服务器响应（可选）
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=0.001)
                        # print(f"收到响应: {response}")
                    except asyncio.TimeoutError:
                        pass
                    
                    packet_count += 1
                    
                    # 统计
                    if packet_count % frequency == 0:
                        elapsed = time.time() - start_time
                        actual_freq = packet_count / elapsed
                        print(f"📊 发送频率: {actual_freq:.1f} Hz")
                    
                    # 控制频率
                    elapsed = time.time() - loop_start
                    if elapsed < dt:
                        await asyncio.sleep(dt - elapsed)
                        
            except KeyboardInterrupt:
                print("\n🛑 停止发送")


def main():
    parser = argparse.ArgumentParser(description="WebSocket 机械臂发送器")
    parser.add_argument('--url', type=str, required=True,
                        help='WebSocket URL (如 ws://192.168.1.100:8765)')
    parser.add_argument('--serial-port', type=str, default=None)
    parser.add_argument('--baudrate', type=int, default=115200)
    parser.add_argument('--frequency', type=int, default=100)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🤖 WebSocket 机械臂控制器")
    print("=" * 60)
    print(f"服务器: {args.url}")
    print(f"频率: {args.frequency} Hz")
    print("=" * 60)
    
    sender = WebSocketArmSender(
        url=args.url,
        serial_port=args.serial_port,
        baudrate=args.baudrate
    )
    
    try:
        asyncio.run(sender.send_loop(frequency=args.frequency))
    except KeyboardInterrupt:
        print("\n✅ 程序已退出")


if __name__ == "__main__":
    main()


















