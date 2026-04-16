#!/usr/bin/env python3
"""
笔抓取多场景配置文件

定义所有场景的笔位置配置，供多场景调参与回放模块使用。

修改笔位置只需编辑这一个文件即可。
"""
import numpy as np

# paper_box 的固定位置（从 scene.xml keyframe 中提取）
BOX_POSITION = np.array([0.075, -0.45, 0.753])
BOX_QUATERNION = [1, 0, 0, 0]  # qw, qx, qy, qz
BOX_QPOS_START = 6  # paper_box 在 qpos 中的起始索引

# 定义多个笔位置场景（同时移动所有4支笔，保持相对位置）
_BOUNDARY_PENS = {
    'pen1': (np.array([0.20, -0.40, 0.756]), [0.7071, 0.7071, 0, 0]),
    'pen_2': (np.array([0.25, -0.33, 0.756]), [0.5, 0.5, 0.5, 0.5]),
    'pen_3': (np.array([0.35, -0.35, 0.756]), [0.7071, 0.7071, 0, 0]),
    'pen_4': (np.array([0.30, -0.43, 0.756]), [0.5, 0.5, -0.5, -0.5]),
}
_BOUNDARY_POS = np.array([pos for pos, _ in _BOUNDARY_PENS.values()])
_X_MIN, _Y_MIN = _BOUNDARY_POS[:, 0].min(), _BOUNDARY_POS[:, 1].min()
_X_MAX, _Y_MAX = _BOUNDARY_POS[:, 0].max(), _BOUNDARY_POS[:, 1].max()
_PEN_HALF_LENGTH = 0.055
_PEN_RADIUS = 0.0042


def _random_pens(seed: int):
    rng = np.random.default_rng(seed)
    z = 0.756
    base_quat = np.array([0.7071, 0.7071, 0.0, 0.0])
    def _quat_mul(a, b):
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return np.array([
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ])
    def yaw_quat(yaw):
        half = yaw / 2.0
        qz = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
        return _quat_mul(qz, base_quat).tolist()
    positions = []
    yaws = []
    for _ in range(4):
        yaws.append(rng.uniform(-np.pi, np.pi))
        positions.append(np.array([rng.uniform(_X_MIN, _X_MAX), rng.uniform(_Y_MIN, _Y_MAX), z]))
    return {
        'pen1': (positions[0], yaw_quat(yaws[0])),
        'pen_2': (positions[1], yaw_quat(yaws[1])),
        'pen_3': (positions[2], yaw_quat(yaws[2])),
        'pen_4': (positions[3], yaw_quat(yaws[3])),
    }


PEN_SCENARIOS = [
    {
        'id': 0,
        'name': 'boundary_test',
        'description': '边界测试',
        'pens': _BOUNDARY_PENS,
    },
    {
        'id': 1,
        'name': 'Original Position',
        'description': '所有笔原始位置',
        'pens': {
            'pen1': (np.array([0.20, -0.40, 0.756]), [0.7071, 0.7071, 0, 0]),
            'pen_2': (np.array([0.25, -0.35, 0.756]), [0.7071, 0.7071, 0, 0]),
            'pen_3': (np.array([0.35, -0.35, 0.756]), [0.7071, 0.7071, 0, 0]),
            'pen_4': (np.array([0.30, -0.35, 0.756]), [0.7071, 0.7071, 0, 0]),
        },
    },
    {
        'id': 2,
        'name': 'scenario2',
        'description': 'scenario2 笔位置',
        'pens': {
            'pen1': (np.array([0.25, -0.33, 0.754]), [0.5, 0.5, 0.5, 0.5]),
            'pen_2': (np.array([0.25, -0.36, 0.756]), [0.5, 0.5, 0.5, 0.5]),
            'pen_3': (np.array([0.35, -0.4, 0.756]), [0.5, 0.5, -0.5, -0.5]),  
            'pen_4': (np.array([0.30, -0.43, 0.756]), [0.5, 0.5, -0.5, -0.5]),
        },
    },
    {
        'id': 3,
        'name': 'scenario3',
        'description': 'scenario3 笔位置',
        'pens': {
            'pen1': (np.array([0.4, -0.4, 0.756]), [0.181, 0.181, 0.684, 0.684]),
            'pen_2': (np.array([0.25, -0.362, 0.756]), [0.545, 0.545, 0.45, 0.45]),
            'pen_3': (np.array([0.248, -0.394, 0.756]), [0.703, 0.703, -0.071, -0.071]),
            'pen_4': (np.array([0.35, -0.4, 0.756]), [0.655, 0.655, -0.266, -0.266]),
        },
    },
    {
        'id': 4,
        'name': 'scenario4',
        'description': 'scenario4 笔位置',
        'pens': {
            'pen1': (np.array([0.35, -0.4, 0.756]), [0.318, 0.318, -0.632, -0.632]),
            'pen_2': (np.array([0.224, -0.34, 0.756]), [0.683, 0.683, -0.183, -0.183]),
            'pen_3': (np.array([0.31, -0.35, 0.756]), [0.545, 0.545, 0.451, 0.451]),
            'pen_4': (np.array([0.245, -0.418, 0.756]), [0.173, 0.173, 0.686, 0.686]),
        },
    },
    {
        'id': 5,
        'name': 'scenario5',
        'description': 'scenario5 笔位置',
        'pens': {
            'pen1': (np.array([0.35, -0.372, 0.756]), [0.137, 0.137, -0.694, -0.694]),
            'pen_2': (np.array([0.24, -0.34, 0.756]), [0.585, 0.585, 0.397, 0.397]),
            'pen_3': (np.array([0.282, -0.39, 0.756]), [0.681, 0.681, 0.192, 0.192]),
            'pen_4': (np.array([0.2, -0.368, 0.756]), [0.016, 0.016, 0.707, 0.707]),
        },
    },
]

# qpos索引映射（MuJoCo场景中的位置）
# qpos[0:6]    → 机械臂的6个关节
# qpos[6:13]   → 纸盒 (paper_box) freejoint
# qpos[13:20]  → pen1 freejoint
# qpos[20:27]  → pen_2 freejoint
# qpos[27:34]  → pen_3 freejoint
# qpos[34:41]  → pen_4 freejoint
PEN_QPOS_MAP = {
    'pen1': 13,
    'pen_2': 20,
    'pen_3': 27,
    'pen_4': 34,
}
