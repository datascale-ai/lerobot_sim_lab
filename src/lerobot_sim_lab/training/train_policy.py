#!/usr/bin/env python3
"""
训练 SO-100 Pick & Place 策略

支持的模型:
- diffusion: Diffusion Policy (推荐，适合机械臂操作任务)
- act: Action Chunking with Transformers (推荐，适合长时序任务)
- tdmpc: Temporal Difference Model Predictive Control
- vqbet: Vector-Quantized Behavior Transformer

用法:
    # 使用 Diffusion Policy 训练
    python3 train_so100_policy.py --policy diffusion --dataset-path ./data/so100_pick_scripted --steps 10000
    
    # 使用 ACT 训练
    python3 train_so100_policy.py --policy act --dataset-path ./data/so100_pick_scripted --steps 10000
"""
import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# 导入 LeRobot 组件（可选依赖）
try:
    from lerobot.configs.types import FeatureType
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.policies.act.configuration_act import ACTConfig

    # 导入不同的 policy 配置
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.tdmpc.configuration_tdmpc import TDMPCConfig
    from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig

    HAS_LEROBOT = True
except ImportError:
    HAS_LEROBOT = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_policy_config(policy_name, input_features, output_features, args):
    """根据policy名称创建对应的配置"""
    if policy_name == "diffusion":
        return DiffusionConfig(
            input_features=input_features,
            output_features=output_features,
            # Diffusion Policy 推荐参数
            n_obs_steps=2,              # 使用2帧观测
            horizon=16,                 # 预测16步动作
            n_action_steps=8,           # 执行前8步
            num_inference_steps=10,     # 推理时的扩散步数
            down_dims=[256, 512, 1024], # 编码器维度
            device=args.device,
        )
    elif policy_name == "act":
        return ACTConfig(
            input_features=input_features,
            output_features=output_features,
            # ACT 推荐参数
            n_obs_steps=1,              # 使用当前帧
            chunk_size=100,             # 预测100步动作块
            n_action_steps=100,         # 执行所有预测的动作
            hidden_dim=512,             # Transformer隐藏层维度
            dim_feedforward=3200,       # 前馈层维度
            n_encoder_layers=4,         # 编码器层数
            n_decoder_layers=7,         # 解码器层数
            device=args.device,
        )
    elif policy_name == "tdmpc":
        return TDMPCConfig(
            input_features=input_features,
            output_features=output_features,
            n_obs_steps=1,
            device=args.device,
        )
    elif policy_name == "vqbet":
        return VQBeTConfig(
            input_features=input_features,
            output_features=output_features,
            n_obs_steps=1,
            device=args.device,
        )
    else:
        raise ValueError(f"Unsupported policy: {policy_name}")


def train(args):
    """主训练函数"""
    logger.info("=" * 80)
    logger.info("训练 SO-100 Pick & Place 策略")
    logger.info(f"  Policy: {args.policy}")
    logger.info(f"  Dataset: {args.dataset_path}")
    logger.info(f"  Training steps: {args.steps}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Device: {args.device}")
    logger.info("=" * 80)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # 加载数据集元数据
    logger.info(f"Loading dataset metadata from {args.dataset_path}...")
    dataset_metadata = LeRobotDatasetMetadata(args.dataset_path)
    
    # 准备 policy 的输入/输出特征
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    
    logger.info(f"Input features: {list(input_features.keys())}")
    logger.info(f"Output features: {list(output_features.keys())}")
    
    # 创建 policy 配置
    logger.info(f"Creating {args.policy} policy configuration...")
    cfg = get_policy_config(args.policy, input_features, output_features, args)
    
    # 创建 policy 模型
    logger.info("Instantiating policy model...")
    policy = make_policy(cfg)
    policy.train()
    policy.to(args.device)
    
    # 计算模型参数量
    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logger.info(f"Policy has {n_params:,} trainable parameters")
    
    # 创建预处理器和后处理器
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)
    
    # 配置 delta_timestamps (根据不同 policy 调整)
    if args.policy == "diffusion":
        delta_timestamps = {
            "observation.state": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
            "observation.images.front": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
            "observation.images.side": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
            "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
        }
    elif args.policy == "act":
        delta_timestamps = {
            "observation.state": [0.0],  # ACT 只用当前帧
            "observation.images.front": [0.0],
            "observation.images.side": [0.0],
            "action": [i / dataset_metadata.fps for i in range(cfg.chunk_size)],  # 预测整个动作块
        }
    else:
        # 其他 policy 的默认配置
        delta_timestamps = {
            "observation.state": [0.0],
            "observation.images.front": [0.0],
            "observation.images.side": [0.0],
            "action": [0.0],
        }
    
    logger.info(f"Delta timestamps: {delta_timestamps}")
    
    # 加载数据集
    logger.info("Loading training dataset...")
    dataset = LeRobotDataset(args.dataset_path, delta_timestamps=delta_timestamps)
    logger.info(f"Dataset size: {len(dataset)} samples")
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=(args.device != "cpu"),
        drop_last=True,
    )
    
    # 训练循环
    logger.info("Starting training...")
    step = 0
    epoch = 0
    done = False
    best_loss = float('inf')
    
    while not done:
        epoch += 1
        logger.info(f"\n{'='*80}")
        logger.info(f"Epoch {epoch}")
        logger.info(f"{'='*80}")
        
        for batch in dataloader:
            # 预处理
            batch = preprocessor(batch)
            
            # 前向传播
            loss, loss_dict = policy.forward(batch)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip_norm)
            
            # 更新参数
            optimizer.step()
            optimizer.zero_grad()
            
            # 日志
            if step % args.log_freq == 0:
                logger.info(f"Step {step:6d} | Loss: {loss.item():.4f}")
                if loss_dict:
                    for key, value in loss_dict.items():
                        logger.info(f"  {key}: {value:.4f}")
            
            # 保存检查点
            if step % args.save_freq == 0 and step > 0:
                checkpoint_dir = output_dir / f"checkpoint_{step}"
                checkpoint_dir.mkdir(exist_ok=True)
                logger.info(f"Saving checkpoint to {checkpoint_dir}")
                policy.save_pretrained(checkpoint_dir)
                preprocessor.save_pretrained(checkpoint_dir)
                postprocessor.save_pretrained(checkpoint_dir)
                
                # 保存最佳模型
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_dir = output_dir / "best"
                    best_dir.mkdir(exist_ok=True)
                    logger.info(f"New best loss: {best_loss:.4f}, saving to {best_dir}")
                    policy.save_pretrained(best_dir)
                    preprocessor.save_pretrained(best_dir)
                    postprocessor.save_pretrained(best_dir)
            
            step += 1
            if step >= args.steps:
                done = True
                break
    
    # 保存最终模型
    final_dir = output_dir / "final"
    final_dir.mkdir(exist_ok=True)
    logger.info(f"\nTraining completed! Saving final model to {final_dir}")
    policy.save_pretrained(final_dir)
    preprocessor.save_pretrained(final_dir)
    postprocessor.save_pretrained(final_dir)
    
    logger.info("=" * 80)
    logger.info(f"Training finished after {step} steps")
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info(f"Models saved to: {output_dir}")
    logger.info("=" * 80)


def main():
    if not HAS_LEROBOT:
        print("Error: lerobot is required for training. Install it with: pip install 'lerobot-sim-lab[lerobot]'")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Train SO-100 Pick & Place Policy")
    
    # 模型和数据集
    parser.add_argument("--policy", type=str, default="diffusion",
                       choices=["diffusion", "act", "tdmpc", "vqbet"],
                       help="Policy type to train")
    parser.add_argument("--dataset-path", type=str, required=True,
                       help="Path to LeRobot dataset (e.g., ./data/so100_pick_scripted)")
    
    # 训练参数
    parser.add_argument("--steps", type=int, default=10000,
                       help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-6,
                       help="Weight decay for optimizer")
    parser.add_argument("--grad-clip-norm", type=float, default=10.0,
                       help="Gradient clipping norm (0 to disable)")
    
    # 数据加载
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of dataloader workers")
    
    # 日志和保存
    parser.add_argument("--output-dir", type=str, default="./outputs/train_so100",
                       help="Output directory for checkpoints")
    parser.add_argument("--log-freq", type=int, default=100,
                       help="Logging frequency (steps)")
    parser.add_argument("--save-freq", type=int, default=1000,
                       help="Checkpoint saving frequency (steps)")
    
    # 设备
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    train(args)


if __name__ == "__main__":
    main()



