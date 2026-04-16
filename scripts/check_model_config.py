#!/usr/bin/env python3
"""
检查预训练模型的配置信息
用于验证模型与环境的兼容性
"""

import sys
from pathlib import Path

def check_model_config(model_path: str):
    """检查模型配置"""
    print("=" * 80)
    print(f"检查模型配置: {model_path}")
    print("=" * 80)
    
    try:
        from lerobot.configs.policies import PreTrainedConfig
        
        print("\n[1/2] 加载配置文件...")
        config = PreTrainedConfig.from_pretrained(model_path)
        
        print(f"✅ 配置加载成功！")
        print(f"\n模型类型: {config.name}")
        
        # 输入特征
        print("\n" + "=" * 80)
        print("输入特征 (Input Features):")
        print("=" * 80)
        if hasattr(config, 'input_features') and config.input_features:
            for key, feature in config.input_features.items():
                print(f"  {key}:")
                if hasattr(feature, 'shape'):
                    print(f"    - shape: {feature.shape}")
                if hasattr(feature, 'feature_type'):
                    print(f"    - type: {feature.feature_type}")
        else:
            print("  (未找到输入特征信息)")
        
        # 输出特征
        print("\n" + "=" * 80)
        print("输出特征 (Output Features):")
        print("=" * 80)
        if hasattr(config, 'output_features') and config.output_features:
            for key, feature in config.output_features.items():
                print(f"  {key}:")
                if hasattr(feature, 'shape'):
                    print(f"    - shape: {feature.shape}")
                if hasattr(feature, 'feature_type'):
                    print(f"    - type: {feature.feature_type}")
        else:
            print("  (未找到输出特征信息)")
        
        # 图像特征
        print("\n" + "=" * 80)
        print("图像配置:")
        print("=" * 80)
        if hasattr(config, 'image_features') and config.image_features:
            print(f"  图像键: {config.image_features}")
        else:
            print("  (无图像输入)")
        
        # 模型超参数
        print("\n" + "=" * 80)
        print("模型超参数:")
        print("=" * 80)
        if hasattr(config, 'chunk_size'):
            print(f"  chunk_size: {config.chunk_size}")
        if hasattr(config, 'n_action_steps'):
            print(f"  n_action_steps: {config.n_action_steps}")
        if hasattr(config, 'dim_model'):
            print(f"  dim_model: {config.dim_model}")
        if hasattr(config, 'use_vae'):
            print(f"  use_vae: {config.use_vae}")
        
        # 设备信息
        print("\n" + "=" * 80)
        print("设备配置:")
        print("=" * 80)
        print(f"  device: {config.device if hasattr(config, 'device') else 'N/A'}")
        
        print("\n" + "=" * 80)
        print("完整配置 (Config Dict):")
        print("=" * 80)
        import json
        # 转换为字典并格式化输出
        config_dict = config.__dict__
        print(json.dumps({k: str(v) for k, v in config_dict.items()}, indent=2))
        
        print("\n" + "=" * 80)
        print("✅ 配置检查完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 配置加载失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="检查预训练模型配置")
    parser.add_argument("model_path", type=str, 
                        help="模型路径（Hub ID 或本地路径）")
    args = parser.parse_args()
    
    check_model_config(args.model_path)

