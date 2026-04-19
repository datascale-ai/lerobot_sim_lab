#!/usr/bin/env bash
# apply_patches.sh — 将 SO-100 补丁应用到 pip 安装的 lerobot 和 gym-hil
#
# 用法:
#   bash patches/apply_patches.sh [--dry-run]
#
# 前提:
#   pip install lerobot==0.4.3 gym-hil==0.1.13
#
# 补丁来源:
#   patches/gym-hil-v0.1.13-so100.patch   — gym-hil SO-100 环境支持
#   patches/lerobot-v0.4.3-so100.patch    — lerobot SO-100 训练/评估适配
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DRY_RUN=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
    echo "[dry-run] 仅检查补丁是否可应用，不做实际修改"
fi

# ---------- 定位 site-packages ----------
PYTHON="${PYTHON:-python3}"
SITE_PACKAGES=$($PYTHON -c "import site; print(site.getsitepackages()[0])")
echo "Python: $($PYTHON --version)"
echo "site-packages: $SITE_PACKAGES"

# ---------- 应用 gym-hil 补丁 ----------
GYM_HIL_DIR="$SITE_PACKAGES/gym_hil"
GYM_HIL_PATCH="$SCRIPT_DIR/gym-hil-v0.1.13-so100.patch"

if [ -d "$GYM_HIL_DIR" ]; then
    echo ""
    echo "========== 应用 gym-hil 补丁 =========="
    echo "目标: $GYM_HIL_DIR"

    # patch -p1 strips "a/" or "b/" prefix; we apply from parent of gym_hil/
    cd "$SITE_PACKAGES"
    if patch -p0 --forward $DRY_RUN < "$GYM_HIL_PATCH"; then
        echo "[OK] gym-hil 补丁应用成功"
    else
        echo "[WARN] gym-hil 补丁已应用或存在冲突，请检查上方输出"
    fi
else
    echo ""
    echo "[SKIP] gym-hil 未安装（未找到 $GYM_HIL_DIR）"
    echo "  安装: pip install gym-hil==0.1.13"
fi

# ---------- 应用 lerobot 补丁 ----------
LEROBOT_DIR="$SITE_PACKAGES/lerobot"
LEROBOT_PATCH="$SCRIPT_DIR/lerobot-v0.4.3-so100.patch"

if [ -d "$LEROBOT_DIR" ]; then
    echo ""
    echo "========== 应用 lerobot 补丁 =========="
    echo "目标: $LEROBOT_DIR"

    LEROBOT_VER="$($PYTHON -c "import importlib.metadata as m; print(m.version('lerobot'))" 2>/dev/null || echo "unknown")"
    if [[ "$LEROBOT_VER" != "0.4.3" ]]; then
        echo ""
        echo "[ERROR] 已安装的 lerobot 版本为 ${LEROBOT_VER}，但补丁仅支持 0.4.3。"
        echo "  请先重装: pip uninstall -y lerobot && pip install 'lerobot==0.4.3'"
        echo "  （若曾部分打补丁失败，请删除 site-packages/lerobot 下 *.rej 后再装）"
        echo "  本仓库可选依赖已固定为 lerobot==0.4.3，请使用: pip install -e \".[lerobot]\""
        exit 1
    fi

    cd "$LEROBOT_DIR"
    if patch -p1 --forward $DRY_RUN < "$LEROBOT_PATCH"; then
        echo "[OK] lerobot 补丁应用成功"
    else
        echo "[WARN] lerobot 补丁已应用或存在冲突，请检查上方输出"
    fi
else
    echo ""
    echo "[SKIP] lerobot 未安装（未找到 $LEROBOT_DIR）"
    echo "  安装: pip install 'lerobot==0.4.3'   # 与补丁版本一致（或按 README 安装可选依赖 [lerobot]）"
fi

echo ""
echo "========== 完成 =========="
echo "验证（本机已激活安装 lerobot_sim_lab 的虚拟环境）："
echo ""
echo "  1) lerobot（补丁目标包，须在 site-packages）："
echo "     $PYTHON -c \"import importlib.metadata as m; assert m.version('lerobot')=='0.4.3'; import lerobot; print('lerobot', m.version('lerobot'), 'OK')\""
echo ""
echo "  2) gym-hil（可选，仅在使用上游 gym_hil 管线时需要）："
echo "     $PYTHON -c \"import gym_hil; print('gym_hil OK')\"   # 未安装则跳过或 pip install gym-hil==0.1.13"
echo ""
echo "  3) 本仓库可编辑安装："
echo "     $PYTHON -c \"import lerobot_sim_lab; lerobot_sim_lab.envs.register_envs(); print('lerobot_sim_lab OK')\""
echo ""
echo "若 (3) 失败：在项目根执行 pip install -e ."
