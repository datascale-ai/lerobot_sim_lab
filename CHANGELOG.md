# Changelog

## 0.1.0 (2026-04-16)

首次开源发布。将原始 SO-100/SO-101 仿真工作区重构为可安装的 Python 包。

### 新增

- `src/lerobot_sim_lab/` 主包，包含 config、envs、data、control、training、evaluation、trajectory、sim、tuning、utils 共 10 个子模块
- `src/gym_hil/` 向后兼容转发层
- Gymnasium 仿真环境：`SO100PickCube-v0`、`SO100PickCubeScripted-v0`、`SO100GrabPen-v0`
- 4 个 CLI 入口：`lerobot-sim-keyboard`、`lerobot-sim-record`、`lerobot-sim-train`、`lerobot-sim-view`
- `patches/` 目录：gym-hil 和 lerobot 的 SO-100 适配补丁及一键应用脚本
- pytest 测试套件（13 个测试）
- 完整文档：quickstart、architecture、contributing、lerobot_patches 等 10 篇

### 重构

- 消除 34 处 `sys.path.append` 硬编码路径
- 删除内嵌的 LeRobot（283 文件）、gym-hil、transformers fork，改用 pip 依赖 + 补丁
- 合并 9 个调优脚本为 4 个（通过 argparse 子命令）
- 统一资产路径管理（支持 `LEROBOT_SIM_LAB_ASSETS` 环境变量覆盖）
