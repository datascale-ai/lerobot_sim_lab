# 第三方 Fork 补丁记录

本项目基于两个第三方库的 fork 修改版本构建。重构时将 fork 代码整合进 `lerobot_sim_lab` 包中，
不再内嵌完整的第三方源码。本文档记录了相对于上游版本的所有自定义修改。

## 补丁文件

补丁保存在 `patches/` 目录下，可通过一键脚本应用到 pip 安装的依赖：

```bash
# 检查补丁是否可应用
bash patches/apply_patches.sh --dry-run

# 应用补丁
bash patches/apply_patches.sh
```

脚本结束时会打印两条验证命令：一条检查 **site-packages 内已打补丁的 `gym_hil` 与 `lerobot`**，另一条检查 **本仓库的 `lerobot_sim_lab` 可编辑安装**。二者含义不同，请勿混淆。

| 文件 | 大小 | 说明 |
|------|------|------|
| [`patches/gym-hil-v0.1.13-so100.patch`](../patches/gym-hil-v0.1.13-so100.patch) | ~1100 行 | gym-hil SO-100 环境支持 |
| [`patches/lerobot-v0.4.3-so100.patch`](../patches/lerobot-v0.4.3-so100.patch) | ~3200 行 | lerobot SO-100 训练/评估适配 |
| [`patches/apply_patches.sh`](../patches/apply_patches.sh) | — | 一键应用脚本 |

---

## 1. gym-hil (huggingface/gym-hil)

- **上游版本**: v0.1.13（与 fork 版本号一致）
- **上游仓库**: https://github.com/huggingface/gym-hil
- **对比基准**: 上游 `main` 分支 HEAD (commit `6ec0078`)
- **处理方式**: 核心代码搬入 `src/lerobot_sim_lab/envs/`

### 1.1 新增文件（上游不存在）

| 文件 | 行数 | 说明 |
|------|------|------|
| `envs/so100_gym_env.py` | 629 | SO-100 抓取方块和抓笔任务的 Gymnasium 环境（`SO100PickCubeGymEnv`, `SO100GrabPenGymEnv`） |
| `envs/so100_pick_scripted_env.py` | 253 | SO-100 脚本化动作序列环境（`SO100PickCubeScriptedEnv`），预定义抓取-放置动作流 |

### 1.2 修改的文件

#### `__init__.py` — 新增 SO-100 环境注册

在原有 Panda 环境注册之后追加了 SO-100 环境 ID 注册（现已统一使用 `lerobot_sim_lab/*` 命名空间）：

```python
# 新增内容（60行）
register(id="lerobot_sim_lab/SO100PickCube-v0", ...)
register(id="lerobot_sim_lab/SO100GrabPen-v0", ...)
register(id="lerobot_sim_lab/SO100PickCubeScripted-v0", ...)
```

#### `envs/__init__.py` — 导出 SO-100 类

```python
# 新增（现位于 lerobot_sim_lab.envs）
from lerobot_sim_lab.envs.so100_gym_env import SO100PickCubeGymEnv, SO100GrabPenGymEnv
from lerobot_sim_lab.envs.so100_scripted_env import SO100PickCubeScriptedEnv
# __all__ 新增 3 个类
```

#### `wrappers/factory.py` — 传递 task 属性

在 `make_env()` 返回前新增 task/task_description 属性传递（7行）：

```python
if hasattr(env, "unwrapped"):
    base_env = env.unwrapped
    if hasattr(base_env, "task"):
        env.task = base_env.task
    if hasattr(base_env, "task_description"):
        env.task_description = base_env.task_description
```

**原因**: LeRobot 评估流程会读取 `env.task_description`，但经过多层 Wrapper 后该属性丢失。

#### `wrappers/viewer_wrapper.py` — 增强 PassiveViewerWrapper

主要修改（约 70 行新增）：

1. **新增构造参数**: `default_camera`（默认相机名）、`lock_camera`（锁定相机）、`sync_every_n_steps`（同步频率）
2. **默认相机视角设置**: 初始化时根据 `default_camera` 参数计算 lookat/distance/azimuth/elevation，支持锁定和自由两种模式
3. **性能优化**: `step()` 中每 N 步才调用 `viewer.sync()`，而非每步都同步，通过 `sync_every_n_steps` 控制

---

## 2. lerobot (huggingface/lerobot)

- **上游版本**: v0.4.3（tag `v0.4.3`）
- **上游仓库**: https://github.com/huggingface/lerobot
- **对比基准**: Git tag `v0.4.3`
- **处理方式**: 重构后删除内嵌源码，改为 pip 依赖 `lerobot==0.4.3`（与补丁一致）

> **注意**: 本地 fork 基于 v0.4.3 之前某个开发版本（缺少 `RobotObservation` 类、缺少 `pi0_fast` 策略等），
> 部分 diff 是上游在 v0.4.3 发布前的变更，而非本地修改。以下仅记录**确认为本地自定义修改**的内容。

### 2.1 新增文件（上游不存在）

| 路径 | 说明 |
|------|------|
| `robots/so100_follower/` | SO-100 单臂 follower 机器人配置和驱动 |
| `robots/so101_follower/` | SO-101 单臂 follower 机器人配置和驱动 |
| `robots/bi_so100_follower/` | SO-100 双臂 follower 机器人配置 |
| `teleoperators/so100_leader/` | SO-100 单臂 leader 遥操作器 |
| `teleoperators/so101_leader/` | SO-101 单臂 leader 遥操作器 |
| `teleoperators/bi_so100_leader/` | SO-100 双臂 leader 遥操作器 |
| `processor/joint_observations_processor.py` | 关节速度和电机电流处理步骤（`JointVelocityProcessorStep`, `MotorCurrentProcessorStep`） |
| `dev_docs/` | 项目内部笔记和总结（22个文件），非运行时代码 |
| `mcp-tools/` | 本地 MCP 工具（6个文件），非运行时代码 |
| `examples/remote_mujoco_teleop/` | 远程 MuJoCo 遥操作示例（6个文件） |

### 2.2 修改的文件（按影响程度排序）

#### `scripts/lerobot_train.py` — 注入 gym-hil 环境支持（+112 行）

最大的修改。在训练脚本中：
1. 通过 `sys.path.insert` 注入 gym-hil 路径
2. 注册 `gym_gym_manipulator/SO100PickCubeBase-v0` 等环境 ID（LeRobot 内部会拼接 `gym_{package_name}/{task}` 格式）
3. 新增 `get_default_peft_configuration()` 函数，为 SmolVLA/Pi0/Pi05 策略提供默认 PEFT 配置

#### `scripts/lerobot_record.py` — SO-100 设备路径和双臂配置（+74 行）

将文档示例和导入中的 `bi_so_follower`/`so_follower`/`so_leader` 重命名为 `bi_so100_follower`/`so100_follower`/`so100_leader`，更新设备串口路径、相机配置等。

#### `scripts/lerobot_teleoperate.py` — 同上重命名（+53 行）

同 `lerobot_record.py`，将所有机器人/遥操作器引用从 `so_*` 改为 `so100_*`/`so101_*`。

#### `envs/utils.py` — gym-hil 兼容（+12 行）

1. 新增对 `observations["state"]` 键的支持（gym-hil 环境使用 `state` 而非 `agent_pos`）
2. 将 `add_envs_task()` 的类型签名从 `RobotObservation` 改为 `dict[str, Any]`（兼容旧版）

#### `rl/gym_manipulator.py` — SO-100 集成（+22 行）

1. 将 `so_follower` → `so100_follower`、`so_leader` → `so101_leader`
2. 将 `RobotObservation` 类型改为 `dict[str, Any]`
3. 从 `processor` 模块引入 `JointVelocityProcessorStep` 和 `MotorCurrentProcessorStep`

#### `policies/smolvla/modeling_smolvla.py` — 移除方法（-19 行）

删除了 `_get_default_peft_targets()` 和 `_validate_peft_config()` 方法。
**原因**: PEFT 配置逻辑被移到了 `lerobot_train.py` 中的 `get_default_peft_configuration()` 全局函数。

#### 其他小修改

| 文件 | 变更行数 | 说明 |
|------|----------|------|
| `robots/utils.py` | +10 | 添加 `so100_follower`/`so101_follower` 注册 |
| `teleoperators/utils.py` | +10 | 添加 `so100_leader`/`so101_leader`/`bi_so100_leader` 注册 |
| `scripts/lerobot_calibrate.py` | +8 | 设备名重命名 |
| `scripts/lerobot_replay.py` | +11 | 设备名重命名 |
| `scripts/lerobot_eval.py` | +24 | 评估脚本适配 |
| `robots/hope_jr/hope_jr_arm.py` | +20 | 小调整 |

### 2.3 当前项目的影响

重构后 `lerobot_sim_lab` 不再内嵌 lerobot 源码，改用 pip 安装的 `lerobot==0.4.3`。这意味着：

1. **`envs/utils.py` 的 `state` 键支持** — 该 HACK 在 pip 版本中不存在。当前 `lerobot_sim_lab` 的环境使用
   `observation["state"]` 键，如果需要与 `lerobot-eval` 对接，可能需要在环境侧将 `state` 映射为 `agent_pos`。
2. **SO-100 机器人驱动** — 上游 v0.4.3 使用 `so_follower` 命名，fork 使用 `so100_follower`。使用 pip 版本时需注意命名差异。
3. **PEFT 配置** — fork 将 PEFT 默认配置从策略类移到训练脚本，pip 版本保留在策略类中。

---

## 3. transformers fork

- **原始位置**: `workspace/transformers-fix-lerobot_openpi/`
- **处理方式**: 直接删除，未提取 patch
- **说明**: 该 fork 用于修复 OpenPI/SmolVLA 工作流中的兼容性问题。重构后通过 pip 版本管理兼容性，详见 [transformers_compatibility.md](transformers_compatibility.md)
