#!/usr/bin/env python3
"""
格式化waypoints.json，让config数组在一行显示
"""
import json
import sys
from pathlib import Path


def format_compact_config(data):
    """手动构建JSON字符串，config在一行"""
    lines = ["{"]
    lines.append('  "waypoints": [')
    
    waypoints = data['waypoints']
    for i, wp in enumerate(waypoints):
        # 开始waypoint对象
        lines.append('    {')
        
        # name字段
        lines.append(f'      "name": "{wp["name"]}",')
        
        # config字段（一行）
        config_str = ', '.join(f'{x:.3g}' if abs(x) < 10 else f'{x:.2f}' for x in wp['config'])
        lines.append(f'      "config": [{config_str}]' + (',' if 'steps' in wp or 'timestamp' in wp else ''))
        
        # 可选的steps字段
        if 'steps' in wp:
            lines.append(f'      "steps": {wp["steps"]}' + (',' if 'timestamp' in wp else ''))
        
        # 可选的timestamp字段
        if 'timestamp' in wp:
            lines.append(f'      "timestamp": {wp["timestamp"]}')
        
        # 结束waypoint对象
        lines.append('    }' + (',' if i < len(waypoints) - 1 else ''))
    
    lines.append('  ]')
    lines.append('}')
    
    return '\n'.join(lines)


def main():
    if len(sys.argv) < 2:
        print("用法: python3 format_waypoints_compact.py <waypoints.json>")
        print("示例: python3 format_waypoints_compact.py pen_grab_tuning/scenario_1/waypoints.json")
        return
    
    file_path = Path(sys.argv[1])
    
    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return
    
    # 读取JSON
    with open(file_path, encoding='utf-8') as f:
        data = json.load(f)
    
    # 格式化
    formatted = format_compact_config(data)
    
    # 保存备份
    backup_path = file_path.with_suffix('.json.bak')
    print(f"💾 备份原文件到: {backup_path}")
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # 保存格式化后的文件
    print(f"✍️  格式化并保存到: {file_path}")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(formatted)
    
    print("✅ 完成！")
    print("\n格式化后的样例：")
    print(formatted.split('\n')[3:7])  # 显示第一个waypoint


if __name__ == "__main__":
    main()

