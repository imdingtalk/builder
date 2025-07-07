FROM debian:11.5

# 设置环境变量
ENV container=docker \
    PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=America/New_York \
    HOME=/home/trader \
    TMPDIR=/home/trader/.tmp

# 创建用户和必要目录
RUN useradd -m -d /home/trader -s /bin/bash -G sudo trader && \
    mkdir -p /var/run/sshd /run/sshd /tmp

WORKDIR /home/trader/mmr

# 安装系统包（合并多个 RUN 命令）
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        dialog apt-utils \
        python3 python3-pip python3-venv \
        git wget vim dpkg \
        build-essential \
        tzdata \
        linux-headers-generic \
        lzip curl \
        locales-all \
        redis-server \
        openssh-server \
        sudo unzip tmux expect \
        iproute2 net-tools rsync \
        iputils-ping lnav \
        # TWS 依赖
        libgtk-3-0 libasound2 libnss3 libgbm1 libnspr4 \
        # pyenv 构建依赖
        libbz2-dev libsqlite3-dev libreadline-dev \
        libedit-dev libncurses5-dev libssl-dev libffi-dev liblzma-dev \
        # TWS JDK 依赖
        libtiff5-dev libtiff5 libjbig-dev \
        libpango-1.0-0 libpangocairo-1.0-0 libcairo2-dev \
        libgdk-pixbuf2.0-0 jq \
        # VNC 和 X11
        tigervnc-scraping-server xvfb xterm && \
    # 设置时区
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    # 添加 MongoDB 仓库并安装
    wget -qO - https://www.mongodb.org/static/pgp/server-5.0.asc | apt-key add - && \
    echo "deb http://repo.mongodb.org/apt/debian bullseye/mongodb-org/5.0 main" | tee /etc/apt/sources.list.d/mongodb-org-5.0.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends mongodb-org && \
    # 清理缓存
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# 设置用户密码
RUN echo 'trader:trader' | chpasswd

# 暴露端口
EXPOSE 22 7496 6379 27017 8081 5900 5901

# 复制源代码
COPY ./ /home/trader/mmr/

# 创建必要目录并下载 IBC
RUN mkdir -p /home/trader/ibc/logs \
             /home/trader/mmr/data/redis \
             /home/trader/mmr/data/mongodb \
             /home/trader/mmr/logs \
             /home/trader/.vnc \
             /home/trader/.config \
             /home/trader/.tmp \
             /home/trader/.cache && \
    # 安装 IBC
    wget https://github.com/IbcAlpha/IBC/releases/download/3.16.2/IBCLinux-3.16.2.zip -P /home/trader/mmr/third_party && \
    unzip -o /home/trader/mmr/third_party/IBCLinux-3.16.2.zip -d /home/trader/mmr/third_party/ibc && \
    rm -f /home/trader/mmr/third_party/IBCLinux-3.16.2.zip && \
    chmod +x /home/trader/mmr/third_party/ibc/*.sh && \
    # 设置 VNC 密码
    echo 'trader' | vncpasswd -f > /home/trader/.vnc/passwd && \
    chmod 600 /home/trader/.vnc/passwd

# 复制配置文件并创建日志文件
COPY ./scripts/installation/.bash_profile /home/trader/
RUN touch /home/trader/.hushlogin \
          /home/trader/mmr/logs/trader_service.log \
          /home/trader/mmr/logs/strategy_service.log

# 切换到 trader 用户
USER trader

# 设置 pyenv 环境变量
ENV PYENV_ROOT=$HOME/.pyenv \
    PATH=$HOME/.pyenv/shims:$HOME/.pyenv/bin:$HOME/.local/bin:$PATH

WORKDIR /home/trader/mmr

# 安装 Python 环境和依赖
RUN curl https://pyenv.run | bash && \
    pyenv install 3.10.11 && \
    pyenv virtualenv 3.10.11 mmr && \
    pyenv rehash && \
    # 安装 poetry
    curl -sSL https://install.python-poetry.org | python3 -

# 安装 Python 包（使用缓存挂载）
RUN --mount=type=cache,target=/home/trader/.cache/pip,uid=1000,gid=1000 \
    pip3 install -r /home/trader/mmr/requirements.txt

# 修改文件所有权
USER root
RUN chown -R trader:trader /home/trader

# 切换工作目录并设置入口点
WORKDIR /home/trader
USER trader

# 使用更好的入口点
ENTRYPOINT ["/bin/bash", "-c", "sudo service ssh restart && tail -f /dev/null"]
