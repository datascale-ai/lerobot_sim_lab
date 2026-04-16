# 使用 NVIDIA 官方提供的 CUDA 开发镜像 (Ubuntu 22.04)
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# 避免安装过程中的交互式弹窗
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

ARG USE_CN_MIRRORS=false

# 配置可选 apt 国内镜像源（加速构建）
RUN if [ "$USE_CN_MIRRORS" = "true" ]; then \
      sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
      sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list; \
    fi

# 1. 安装系统级依赖
# ### 修改：在原有基础上增加了 xfce4, tigervnc 等桌面环境依赖
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    wget \
    vim \
    curl \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    patchelf \
    ffmpeg \
    tzdata \
    # --- VNC & Desktop 依赖开始 ---
    xfce4 \
    xfce4-goodies \
    tigervnc-standalone-server \
    x11-xserver-utils \
    dbus-x11 \
    xauth \
    pm-utils \
    upower \
    policykit-1 \
    # --- VNC & Desktop 依赖结束 ---
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

# 配置 pip，并可选启用国内镜像源
RUN pip3 install --upgrade pip && \
    if [ "$USE_CN_MIRRORS" = "true" ]; then \
      pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple; \
    fi

# 2. 设置环境变量
# ### 注意：默认保留 EGL 用于无头训练。但在 VNC 模式下启动脚本会自动修改它。
ENV MUJOCO_GL=egl

# 设置 PyTorch 不占用全部显存
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# ### 新增：设置 VNC 默认显示端口
ENV DISPLAY=:1

# 3. 安装 Python 核心库
RUN pip3 install --no-cache-dir \
    mujoco \
    torch torchvision torchaudio \
    jupyterlab \
    mediapy \
    matplotlib \
    numpy \
    ipywidgets \
    notebook \
    ipympl \
    pyserial \
    websockets \
    zmq

# 启用 JupyterLab 插件
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager || true

# 4. 设置工作目录
WORKDIR /app

# 5. 暴露端口
# 8888: Jupyter, 5901: VNC (Display :1), 9999: 机械臂数据接收, 8765: WebSocket
EXPOSE 8888 5901 9999 8765

# 6. 创建启动脚本（大幅增强）
# ### 修改：现在支持 'jupyter' 和 'vnc' 两种启动模式
RUN echo '#!/bin/bash\n\
\n\
# 函数：启动 Jupyter\n\
start_jupyter() {\n\
    echo "Starting JupyterLab..."\n\
    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \\\n\
        --ServerApp.token="" --ServerApp.password="" \\\n\
        --ServerApp.allow_origin="*" --ServerApp.disable_check_xsrf=True\n\
}\n\
\n\
# 函数：启动 VNC\n\
start_vnc() {\n\
    echo "Starting VNC Desktop..."\n\
    export USER=root\n\
    export HOME=/root\n\
    export MUJOCO_GL=glfw\n\
    \n\
    # 0. 启动系统服务（减少警告）\n\
    echo "Starting system services..."\n\
    mkdir -p /var/run/dbus\n\
    dbus-daemon --system --fork 2>/dev/null || true\n\
    \n\
    # 1. 暴力清理旧锁文件 (解决 "X server already running")\n\
    rm -rf /tmp/.X1-lock /tmp/.X11-unix/X1 /root/.vnc/*.pid /root/.vnc/*.log\n\
    \n\
    # 2. 设置密码\n\
    mkdir -p /root/.vnc\n\
    echo "mujoco" | vncpasswd -f > /root/.vnc/passwd\n\
    chmod 600 /root/.vnc/passwd\n\
    \n\
    # 3. 配置 xstartup（启动 D-Bus 会话并禁用不需要的插件）\n\
    cat > /root/.vnc/xstartup << "XSTARTUP_EOF"\n\
#!/bin/sh\n\
unset SESSION_MANAGER\n\
unset DBUS_SESSION_BUS_ADDRESS\n\
\n\
# 启动 D-Bus 会话\n\
if test -z "$DBUS_SESSION_BUS_ADDRESS"; then\n\
    eval `dbus-launch --sh-syntax --exit-with-session`\n\
    export DBUS_SESSION_BUS_ADDRESS\n\
fi\n\
\n\
# 禁用 XFCE 电源管理插件\n\
xfconf-query -c xfce4-session -p /startup/screensaver/enabled -n -t bool -s false 2>/dev/null || true\n\
\n\
# 启动 XFCE 并重定向警告到日志文件\n\
startxfce4 2>&1 | grep -v "libupower-glib\\|up_client\\|up_device\\|GLib-GObject-CRITICAL\\|pm-is-supported" || true &\n\
\n\
wait\n\
XSTARTUP_EOF\n\
    \n\
    chmod +x /root/.vnc/xstartup\n\
    \n\
    # 4. 启动 VNC Server\n\
    echo "Starting TigerVNC Server..."\n\
    tigervncserver :1 -geometry 1280x800 -depth 24 -localhost no 2>&1 | head -n 5\n\
    \n\
    echo "-------------------------------------------------------"\n\
    echo "✅ VNC Started on port 5901 (Display :1)"\n\
    echo "🔑 Password: mujoco"\n\
    echo "📝 Warnings filtered for cleaner output"\n\
    echo "-------------------------------------------------------"\n\
    \n\
    # 5. 等待并过滤日志输出\n\
    sleep 2\n\
    echo "Desktop environment starting... (filtered logs)"\n\
    tail -f /root/.vnc/*.log 2>/dev/null | grep -v "libupower-glib\\|up_client\\|up_device\\|GLib-GObject\\|pm-is-supported\\|libactions-WARNING" || true\n\
}\n\
\n\
if [ "$1" = "jupyter" ]; then\n\
    start_jupyter\n\
elif [ "$1" = "vnc" ]; then\n\
    start_vnc\n\
else\n\
    exec "$@"\n\
fi' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

# 7. 默认命令
# 你可以在 docker run 时改为 "vnc"
CMD ["jupyter"]
