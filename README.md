# lerobot-sim-lab

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-%3E%3D3.10-blue.svg)](https://www.python.org/)

基于 [LeRobot](https://github.com/huggingface/lerobot) 和 [MuJoCo](https://mujoco.org/) 的具身智能合成数据研究工具包。

`lerobot-sim-lab` 将原始的 SO-100 / SO-101 仿真工作区重构为一个结构清晰、可安装的 Python 包，涵盖仿真环境、数据采集、控制、轨迹规划、策略训练与评估等模块。

## 功能特性

- **Gymnasium 仿真环境** — SO-100 抓取方块、抓笔、脚本化任务等环境，完整支持 Gymnasium API
- **数据管道** — Episode 录制、HDF5 存储、LeRobot 数据集格式转换，支持多相机视角  
- **交互式调优** — 场景、控制、抓笔等交互式调优工具，先把环境与控制调到合适状态再批量录制，提升数据质量
- **多种控制方式** — 键盘、手柄、交互式 GUI、远程遥操作（TCP/WebSocket）
- **轨迹规划** — 基于 OMPL 的多样化轨迹生成（通过 MPLib），支持种子复现
- **策略训练**（两条路径，按需选用）  
  - **`lerobot-sim-train`** — 本仓库提供的轻量入口，直接调用 LeRobot 内置策略配置：**Diffusion Policy**、**ACT**、**TDMPC**、**VQBet**（与上游 `lerobot.policies.*` 对应，实现见 [`training/train_policy.py`](src/lerobot_sim_lab/training/train_policy.py)）  
  - **官方 `lerobot_train.py` 管线** — 与 Hugging Face 生态中更常见的 **SmolVLA**、**Pi0** 等大模型策略一起使用；需安装并打补丁后，参考 [`scripts/train_smolvla.sh`](scripts/train_smolvla.sh)、[`scripts/train_pi.sh`](scripts/train_pi.sh) 等示例脚本
- **策略评估** — 随机/预训练策略评估、仿真与真实对比、SmolVLA 等模型评估（见 `evaluation/`）
- **人机协作（HIL）** — 支持在 RL 训练过程中人类干预的环境 Wrapper（键盘/手柄控制器）
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

# 控制仿真 SO-100（默认键盘；可用 --mode remote / send）
lerobot-sim-control --scenario 0
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

## 数据采集

导出 **LeRobot** 数据集前，请先完成 **`pip install -e ".[lerobot]"`** 以及对 **lerobot 0.4.3** 的补丁步骤（见上文）。录制时会用到 MuJoCo 的离屏渲染；若在纯 SSH、没有显示器的环境运行，请提前配置好 OpenGL（例如设置 **`MUJOCO_GL`**）或使用虚拟显示，以免渲染初始化失败。

### 抓取方块（一键脚本录制）

抓方块是最省事的路径：装好依赖后，直接运行 **`lerobot-sim-record`**（默认 **`scripted`**）。程序会驱动脚本化的抓方块环境，按场景轮换自动跑 episode，并把图像与状态写进 LeRobot 数据集。

```bash
lerobot-sim-record --help
lerobot-sim-record
# 可按需调整：--num-episodes、--repo-id、--root、--fps、--task 等
```

各字段含义与实现细节见 [`src/lerobot_sim_lab/data/record_episodes.py`](src/lerobot_sim_lab/data/record_episodes.py)。

### 抓笔（调轨迹，再导出数据集）

抓笔任务更依赖「先调好场景与轨迹，再批量生成数据」。推荐顺序是：在 **`tuning`** 里把关键帧和回放跑通 → 用 **`trajectory`** 生成或整理关节轨迹 → 最后用 **`lerobot-sim-record --mode trajectory`** 把轨迹批量转成 LeRobot 格式。这样你可以反复改笔位、场景，再统一导出，而不必和抓方块共用同一套脚本入口。

1. **现场调参、看回放**  
   使用 **`python -m lerobot_sim_lab.tuning.tune_pen_grab`**（单场景）或 **`...tune_pen_grab_multi --scenario <编号>`**（多场景）。常用子命令是 **`live`**（看仿真）、**`control`**（存关键帧）、**`playback`**（回放插值轨迹）。若在 **`playback`** 时加上 **`--record`**，会额外存一段 **MP4** 视频，方便肉眼看效果；它**不是** LeRobot 数据集，只是辅助检查。

2. **轨迹文件放哪**  
   默认在仓库根下的 **`pen_grab_tuning/scenario_<编号>/`**：里面有 **`trajectories/episode_XXX.npz`**、**`seed.txt`** 等。需要批量规划时，可看 **`trajectory/`** 里的工具（例如 [`trajectory/generator.py`](src/lerobot_sim_lab/trajectory/generator.py)）。各命令的详细说明写在对应源码文件开头的注释里（如 [`tuning/tune_pen_grab.py`](src/lerobot_sim_lab/tuning/tune_pen_grab.py)）。

3. **转成 LeRobot 数据集**  
   轨迹准备好以后，用 **`trajectory`** 模式一次性写入数据集，例如：

```bash
lerobot-sim-record --mode trajectory \
  --repo-id <你的数据集名或 repo_id> \
  --root ./data \
  --trajectory-root pen_grab_tuning \
  --scenarios "1,2,3,4,5" \
  --seed-file seed.txt
```

训练、评估时仍可使用 Gym 环境 **`lerobot_sim_lab/SO100GrabPen-v0`**；若要做**抓笔的仿真 LeRobot 数据**，请按上面步骤用 **`trajectory` 模式**导出。

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

应用完成后建议自检：

- **补丁目标包**（`site-packages` 内的官方 `gym_hil` / `lerobot`，与仓库内 `src/lerobot_sim_lab` 无关）：`python -c "import gym_hil, lerobot; print('patches OK')"`
- **本仓库包**：`python -c "import lerobot_sim_lab; lerobot_sim_lab.envs.register_envs(); print('lerobot_sim_lab OK')"`

`scripts/` 下的示例 shell 默认以**本仓库根目录**为基准（脚本内自动 `cd` 到仓库根）：数据集多在 `data_grab_pen/`、预训练权重可放在 `third_party/`（该目录已加入 `.gitignore`，需自行下载或软链）。训练脚本默认通过已安装的 `lerobot` 包解析 `lerobot_train.py` 路径；若你使用本地 LeRobot 源码树，可设置环境变量 `LEROBOT_TRAIN` 或 `LEROOT` 覆盖，详见各脚本文件头注释。

完整的修改清单和技术细节参见 [docs/lerobot_patches.md](docs/lerobot_patches.md)。

## 命令行工具

安装后提供以下四个命令行入口：

| 命令 | 功能 |
|------|------|
| `lerobot-sim-control` | SO-100 仿真臂：键盘、`--mode remote` 网络遥操作、`--mode send` 同步真机 |
| `lerobot-sim-record` | 抓方块：默认一键录制；抓笔：用 **`--mode trajectory`** 把 `pen_grab_tuning/` 里的轨迹转成数据集，详见 [数据采集](#数据采集) |
| `lerobot-sim-train` | 轻量训练：LeRobot 内置 Diffusion / ACT / TDMPC / VQBet（SmolVLA、Pi0 等见 `scripts/` + 官方训练脚本） |
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
    control/                 arm_control CLI（键盘/远程/真机）、交互控制、关节探索
      remote/                TCP/WebSocket 远程遥操作
    training/                策略训练（LeRobot 集成）
    evaluation/              策略评估与仿真-真实对比
    trajectory/              轨迹生成、回放与分析
    sim/                     MuJoCo 仿真器封装、场景查看、相机调试
    tuning/                  任务专用参数调优
    utils/                   路径管理、渲染、格式化工具
patches/                     第三方库 SO-100 适配补丁
assets/
  robots/so100/              SO-100 MuJoCo XML、URDF、SRDF、STL 网格
  robots/so101/              SO-101 机器人模型文件
  configs/                   录制配置、远程控制配置
scripts/                     可运行的 Demo 与 Shell 脚本
third_party/                 本地预训练权重等（仅 .gitkeep 入库，其余见 .gitignore）
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
