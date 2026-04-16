"""
本地机械臂数据发送器 - Socket 版本
在本地电脑运行，读取真实机械臂数据并发送到远程 MuJoCo 仿真
"""

import argparse
import json
import socket
import time

import numpy as np

# 串口通信库（根据需要安装：pip install pyserial）
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    print("⚠️  pyserial 未安装，将使用模拟数据")
    SERIAL_AVAILABLE = False


class ArmSender:
    def __init__(self, host, port=9999, serial_port=None, baudrate=115200, arm_type="generic"):
        """
        初始化机械臂发送器
        
        Args:
            host: 远程服务器 IP
            port: 远程服务器端口
            serial_port: 串口端口 (如 'COM3' 或 '/dev/ttyUSB0')
            baudrate: 串口波特率
            arm_type: 机械臂类型 ('dynamixel', 'feetech', 'generic', 'simulator')
        """
        self.host = host
        self.port = port
        self.arm_type = arm_type
        
        # 连接串口（如果提供）
        self.serial_conn = None
        if serial_port and SERIAL_AVAILABLE:
            try:
                self.serial_conn = serial.Serial(serial_port, baudrate, timeout=0.1)
                print(f"✅ 串口已连接: {serial_port} @ {baudrate} baud")
            except Exception as e:
                print(f"❌ 串口连接失败: {e}")
                print("💡 将使用模拟数据模式")
        
        # 模拟模式参数
        self.sim_time = 0.0
        self.num_joints = 6
        
        # 连接到远程服务器
        self.sock = None
        
    def connect(self):
        """连接到远程服务器"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            print(f"✅ 已连接到远程服务器: {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            return False
    
    def read_arm_state(self):
        """
        读取机械臂状态（关节角度和速度）
        
        Returns:
            dict: {'qpos': [...], 'qvel': [...], 'timestamp': ...}
        """
        if self.serial_conn:
            # TODO: 根据您的机械臂协议实现
            # 这里是示例框架
            return self._read_from_serial()
        else:
            # 模拟数据（正弦波运动）
            return self._generate_simulated_data()
    
    def _read_from_serial(self):
        """从串口读取真实机械臂数据"""
        # 示例：Dynamixel 协议
        # 您需要根据实际机械臂的通信协议修改
        try:
            # 发送读取命令
            # self.serial_conn.write(b'read_position\n')
            
            # 读取响应
            # response = self.serial_conn.readline()
            # angles = parse_response(response)
            
            # 临时返回模拟数据
            return self._generate_simulated_data()
            
        except Exception as e:
            print(f"⚠️  串口读取错误: {e}")
            return self._generate_simulated_data()
    
    def _generate_simulated_data(self):
        """生成模拟的机械臂运动数据（用于测试）"""
        self.sim_time += 0.01
        
        # 生成正弦波运动
        qpos = [
            0.5 * np.sin(self.sim_time * 0.5),  # 关节1
            0.3 * np.sin(self.sim_time * 0.8),  # 关节2
            0.4 * np.sin(self.sim_time * 1.0),  # 关节3
            0.2 * np.sin(self.sim_time * 1.2),  # 关节4
            0.3 * np.sin(self.sim_time * 0.9),  # 关节5
            0.5 * np.sin(self.sim_time * 0.7),  # 关节6
        ]
        
        qvel = [
            0.5 * 0.5 * np.cos(self.sim_time * 0.5),
            0.3 * 0.8 * np.cos(self.sim_time * 0.8),
            0.4 * 1.0 * np.cos(self.sim_time * 1.0),
            0.2 * 1.2 * np.cos(self.sim_time * 1.2),
            0.3 * 0.9 * np.cos(self.sim_time * 0.9),
            0.5 * 0.7 * np.cos(self.sim_time * 0.7),
        ]
        
        return {
            'qpos': qpos,
            'qvel': qvel,
            'timestamp': time.time()
        }
    
    def send_data(self, data):
        """发送数据到远程服务器"""
        try:
            message = json.dumps(data) + '\n'
            self.sock.send(message.encode('utf-8'))
            return True
        except Exception as e:
            print(f"❌ 发送失败: {e}")
            return False
    
    def run(self, frequency=100):
        """
        主循环：持续读取并发送数据
        
        Args:
            frequency: 发送频率 (Hz)
        """
        if not self.connect():
            return
        
        dt = 1.0 / frequency
        packet_count = 0
        start_time = time.time()
        last_print = start_time
        
        print(f"🚀 开始发送数据 @ {frequency} Hz")
        print("💡 按 Ctrl+C 停止")
        
        try:
            while True:
                loop_start = time.time()
                
                # 读取机械臂状态
                arm_data = self.read_arm_state()
                
                # 发送数据
                if self.send_data(arm_data):
                    packet_count += 1
                
                # 打印统计信息
                if time.time() - last_print > 1.0:
                    elapsed = time.time() - start_time
                    actual_freq = packet_count / elapsed
                    print(f"📊 发送频率: {actual_freq:.1f} Hz | 总包数: {packet_count}")
                    last_print = time.time()
                
                # 控制频率
                elapsed = time.time() - loop_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                    
        except KeyboardInterrupt:
            print("\n🛑 停止发送")
        finally:
            if self.sock:
                self.sock.close()
            if self.serial_conn:
                self.serial_conn.close()


def main():
    parser = argparse.ArgumentParser(description="机械臂数据发送器")
    parser.add_argument('--host', type=str, required=True,
                        help='远程服务器 IP 地址')
    parser.add_argument('--port', type=int, default=9999,
                        help='远程服务器端口')
    parser.add_argument('--serial-port', type=str, default=None,
                        help='串口端口 (如 COM3 或 /dev/ttyUSB0)')
    parser.add_argument('--baudrate', type=int, default=115200,
                        help='串口波特率')
    parser.add_argument('--frequency', type=int, default=100,
                        help='发送频率 (Hz)')
    parser.add_argument('--arm-type', type=str, default='generic',
                        choices=['dynamixel', 'feetech', 'generic', 'simulator'],
                        help='机械臂类型')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🤖 本地机械臂控制器")
    print("=" * 60)
    print(f"远程地址: {args.host}:{args.port}")
    print(f"机械臂类型: {args.arm_type}")
    if args.serial_port:
        print(f"串口: {args.serial_port} @ {args.baudrate} baud")
    else:
        print("模式: 模拟数据（用于测试）")
    print("=" * 60)
    
    sender = ArmSender(
        host=args.host,
        port=args.port,
        serial_port=args.serial_port,
        baudrate=args.baudrate,
        arm_type=args.arm_type
    )
    
    sender.run(frequency=args.frequency)


if __name__ == "__main__":
    main()


















