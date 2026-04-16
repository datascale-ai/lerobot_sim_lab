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

    cd "$LEROBOT_DIR"
    if patch -p1 --forward $DRY_RUN < "$LEROBOT_PATCH"; then
        echo "[OK] lerobot 补丁应用成功"
    else
        echo "[WARN] lerobot 补丁已应用或存在冲突，请检查上方输出"
    fi
else
    echo ""
    echo "[SKIP] lerobot 未安装（未找到 $LEROBOT_DIR）"
    echo "  安装: pip install 'lerobot-sim-lab[lerobot]'"
fi

echo ""
echo "========== 完成 =========="
echo "验证: python -c \"import gym_hil; import lerobot; print('OK')\""
