import mplib
import numpy as np
import os

print("=" * 50)
print("MPlib 基础测试")
print("=" * 50)

# 1. 检查 mplib 版本和基本信息
print(f"\n📦 MPlib 版本: {mplib.__version__ if hasattr(mplib, '__version__') else '未知'}")
print(f"📦 NumPy 版本: {np.__version__}")

# 2. 检查 URDF 文件是否存在
urdf_path = "so100_sim/so100_6dof/so100.urdf"
print(f"\n📄 URDF 文件: {urdf_path}")
print(f"   存在: {os.path.exists(urdf_path)}")
if os.path.exists(urdf_path):
    print(f"   大小: {os.path.getsize(urdf_path)} 字节")

# 3. 尝试简单初始化
print("\n🔧 尝试初始化 Planner...")
try:
    # move_group 是末端执行器链接的名称
    # 注意：mplib 将 URDF 中的 "Moving_Jaw" 解析为 "Moving Jaw"（空格）
    planner = mplib.Planner(
        urdf=urdf_path,
        move_group="Moving Jaw",  # 注意是空格，不是下划线
        srdf="so100_sim/so100_6dof/so100_mplib.srdf",  # 使用自动生成的 SRDF
        verbose=True  # 显示详细信息
    )
    print("✅ Planner 初始化成功！")
    
    # 获取基本信息
    print(f"\n📊 机器人信息:")
    print(f"   自由度 (DOF): {len(planner.move_group_joint_indices)}")
    print(f"   移动组关节索引: {planner.move_group_joint_indices}")
    print(f"   关节限位 (弧度):")
    for i, limits in enumerate(planner.joint_limits):
        print(f"      关节 {i}: [{limits[0]:.3f}, {limits[1]:.3f}]")
    
    print(f"\n🔧 Planner 可用方法:")
    planner_methods = [m for m in dir(planner) if not m.startswith('_') and callable(getattr(planner, m))]
    for method in sorted(planner_methods):
        print(f"   - {method}")
    
    print(f"\n📋 包含'plan'的方法:")
    plan_methods = [m for m in planner_methods if 'plan' in m.lower()]
    for method in plan_methods:
        print(f"   - {method}")
        import inspect
        try:
            sig = inspect.signature(getattr(planner, method))
            print(f"     签名: {method}{sig}")
        except:
            pass
    
    # 测试碰撞检测
    print(f"\n🔍 碰撞检测测试:")
    init_qpos = np.zeros(len(planner.move_group_joint_indices))
    
    # 使用 planner 的碰撞检测方法
    is_colliding = planner.check_for_self_collision(init_qpos)
    if not is_colliding:
        print("   ✅ 初始姿态（全零）无自碰撞")
    else:
        print("   ⚠️ 初始姿态检测到自碰撞")
    
    # 测试其他姿态
    home_qpos = np.array([0, -np.pi, np.pi, 0.817, 0, -0.157])
    is_colliding_home = planner.check_for_self_collision(home_qpos)
    if not is_colliding_home:
        print("   ✅ Home 姿态无自碰撞")
    else:
        print("   ⚠️ Home 姿态检测到自碰撞")
    
    print(f"\n🎉 MPlib 验证成功！")
    print(f"   - URDF 解析正常")
    print(f"   - SRDF 碰撞对已加载")
    print(f"   - 碰撞检测功能正常")
    print(f"   - 可以开始使用 IK 和轨迹规划功能")
    
except TypeError as e:
    print(f"❌ 参数错误: {e}")
    print("\n💡 尝试查看 Planner 的参数签名...")
    import inspect
    sig = inspect.signature(mplib.Planner.__init__)
    print(f"   Planner.__init__ 参数: {sig}")
    
except Exception as e:
    print(f"❌ 初始化失败: {type(e).__name__}: {e}")
    import traceback
    print("\n📋 详细错误信息:")
    traceback.print_exc()