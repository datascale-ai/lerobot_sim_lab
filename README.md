# lerobot-sim-lab

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-%3E%3D3.10-blue.svg)](https://www.python.org/)

基于 [LeRobot](https://github.com/huggingface/lerobot) 和 [MuJoCo](https://mujoco.org/) 的具身智能合成数据研究工具包。

`lerobot-sim-lab` 将原始的 SO-100 / SO-101 仿真工作区重构为一个结构清晰、可安装的 Python 包，涵盖仿真环境、数据采集、控制、轨迹规划、策略训练与评估等模块。

## 功能特性

- **Gymnasium 仿真环境** — SO-100 抓取方块、抓笔、脚本化任务等环境，完整支持 Gymnasium API
- **数据管道** — Episode 录制、HDF5 存储、LeRobot 数据集格式转换，支持多相机视角
- **多种控制方式** — 键盘、手柄、交互式 GUI、远程遥操作（TCP/WebSocket）
- **轨迹规划** — 基于 OMPL 的多样化轨迹生成（通过 MPLib），支持种子复现
- **策略训练** — 集成 LeRobot，支持 Diffusion Policy、ACT、TDMPC、VQBet 四种策略
- **策略评估** — 随机/预训练策略评估、仿真与真实对比、SmoLVLA 模型评估
- **人机协作（HIL）** — 支持在 RL 训练过程中人类干预的环境 Wrapper（键盘/手柄控制器）
- **交互式调优** — 场景参数、控制参数、抓笔任务等专用调优工具
- **Docker 支持** — 容器化开发环境，可选国内镜像加速

## 快速开始

### 第 1 步：安装基础包

```bash
# 创建并激活虚拟环境（需要 Python >= 3.10）
python -m venv .venv
source .venv/bin/activate

# 安装基础包（仿真环境、数据采集、控制等）
pip install -e .
```

基础包安装完成后即可运行仿真环境：

```bash
# 查看 MuJoCo 场景
lerobot-sim-view

# 键盘控制机器人
lerobot-sim-keyboard
```

### 第 2 步：安装训练/评估依赖（可选）

如需使用 LeRobot 进行策略训练和评估，需要额外安装 `lerobot` 并应用 SO-100 适配补丁：

```bash
# 安装 lerobot 集成
pip install -e ".[lerobot]"

# 应用 SO-100 补丁到 lerobot（必需，否则训练/评估会报错）
bash patches/apply_patches.sh
```

> **为什么需要补丁？** 本项目基于 `lerobot==0.4.3` 的 fork 修改版本开发，新增了
> SO-100 机器人驱动、gym-hil 环境兼容等功能。这些修改尚未合入上游，因此需要通过补丁
> 应用到 pip 安装的版本上。详见 [docs/lerobot_patches.md](docs/lerobot_patches.md)。

### 第 3 步：验证安装

```bash
# 验证基础环境
python -c "
import lerobot_sim_lab.envs
lerobot_sim_lab.envs.register_envs()
import gymnasium as gym
env = gym.make('lerobot_sim_lab/SO100PickCube-v0', render_mode='rgb_array')
obs, info = env.reset(seed=42)
print(f'环境创建成功，观测键: {list(obs.keys())}')
env.close()
"

# 验证 LeRobot 集成（需要先完成第 2 步）
python -c "from lerobot.robots.so100_follower import SO100Follower; print('LeRobot SO-100 补丁已生效')"
```

### 其他可选依赖

```bash
# MPLib 运动规划
pip install -e ".[motion-planning]"

# 远程机械臂控制
pip install -e ".[remote]"

# Jupyter Notebook 支持
pip install -e ".[jupyter]"

# 开发工具（ruff、pytest、mypy 等）
pip install -e ".[dev]"

# 安装全部依赖
pip install -e ".[all]"
```

详细安装说明参见 [docs/quickstart.md](docs/quickstart.md) 和 [docker/README.md](docker/README.md)。

## 补丁说明

本项目依赖两个第三方库的 fork 修改版本。重构后不再内嵌完整的第三方源码，改为通过补丁文件管理差异：

```text
patches/
  gym-hil-v0.1.13-so100.patch    # gym-hil 补丁：SO-100 环境、Viewer 增强
  lerobot-v0.4.3-so100.patch     # lerobot 补丁：SO-100 驱动、训练/评估适配
  apply_patches.sh               # 一键应用脚本
```

| 补丁 | 上游版本 | 主要修改 |
|------|----------|----------|
| `gym-hil` | [v0.1.13](https://github.com/huggingface/gym-hil) | 新增 SO-100 环境（882行），增强 PassiveViewerWrapper（默认相机、性能优化） |
| `lerobot` | [v0.4.3](https://github.com/huggingface/lerobot/tree/v0.4.3) | 新增 SO-100/SO-101 机器人驱动，训练脚本 gym-hil 注入，评估 state 键兼容 |

```bash
# 检查补丁能否正常应用（不做实际修改）
bash patches/apply_patches.sh --dry-run

# 实际应用补丁
bash patches/apply_patches.sh
```

完整的修改清单和技术细节参见 [docs/lerobot_patches.md](docs/lerobot_patches.md)。

## 命令行工具

安装后提供以下四个命令行入口：

| 命令 | 功能 |
|------|------|
| `lerobot-sim-keyboard` | 键盘控制机器人关节 |
| `lerobot-sim-record` | 录制仿真 Episode 并转换为 LeRobot 格式 |
| `lerobot-sim-train` | 策略训练（Diffusion、ACT、TDMPC、VQBet） |
| `lerobot-sim-view` | MuJoCo 场景可视化 |

## 项目结构

```text
src/
  lerobot_sim_lab/           主 Python 包
    config/                  全局常量与场景定义
      scenarios/             抓取方块、抓笔场景配置
    envs/                    Gymnasium 仿真环境
      controllers/           操作空间（末端执行器）控制器
      wrappers/              动作映射、夹爪惩罚、可视化、人机协作 Wrapper
    data/                    Episode 录制与数据集转换
    control/                 键盘控制、交互控制、关节探索
      remote/                TCP/WebSocket 远程遥操作
    training/                策略训练（LeRobot 集成）
    evaluation/              策略评估与仿真-真实对比
    trajectory/              轨迹生成、回放与分析
    sim/                     MuJoCo 仿真器封装、场景查看、相机调试
    tuning/                  任务专用参数调优
    utils/                   路径管理、渲染、格式化工具
  gym_hil/                   gym-hil 向后兼容层
patches/                     第三方库 SO-100 适配补丁
assets/
  robots/so100/              SO-100 MuJoCo XML、URDF、SRDF、STL 网格
  robots/so101/              SO-101 机器人模型文件
  configs/                   录制配置、远程控制配置
scripts/                     可运行的 Demo 与 Shell 脚本
tests/                       Pytest 测试套件
docs/                        项目文档
docker/                      Docker 构建与 DevContainer 文件
examples/notebooks/          Jupyter Notebook 示例
```

## 支持的环境

| 环境 ID | 说明 |
|---------|------|
| `lerobot_sim_lab/SO100PickCube-v0` | SO-100 方块抓取与放置 |
| `lerobot_sim_lab/SO100PickCubeScripted-v0` | SO-100 脚本化抓取与放置 |
| `lerobot_sim_lab/SO100GrabPen-v0` | SO-100 抓笔任务 |

为向后兼容，同时注册了 `gym_hil/*` 格式的环境 ID。

```python
import gymnasium as gym
import lerobot_sim_lab.envs

lerobot_sim_lab.envs.register_envs()

env = gym.make("lerobot_sim_lab/SO100PickCube-v0", render_mode="rgb_array")
obs, info = env.reset(seed=42)

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

## 文档

- [快速开始](docs/quickstart.md)
- [架构概览](docs/architecture.md)
- [贡献指南](docs/contributing.md)
- [第三方补丁说明](docs/lerobot_patches.md)
- [OMPL 安装指南](docs/ompl_install.md)
- [远程机械臂控制](docs/remote_arm_control.md)
- [Docker 部署](docs/docker_setup.md)
- [Transformers 兼容性](docs/transformers_compatibility.md)
- [研究报告（中文）](docs/research_report_cn.md)

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 代码检查
ruff check src/ tests/

# 类型检查
mypy src/lerobot_sim_lab/

# 运行测试
pytest

# 测试覆盖率
pytest --cov=lerobot_sim_lab
```

## 开源协议

[Apache-2.0](LICENSE)
