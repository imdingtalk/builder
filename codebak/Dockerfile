# Use Ubuntu 18.04 as the base image
FROM ubuntu:18.04

# Set environment variables to non-interactive to avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    libssl-dev \
    zlib1g-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libgdbm-dev \
    libdb5.3-dev \
    libbz2-dev \
    libexpat1-dev \
    liblzma-dev \
    tk-dev \
    libffi-dev \
    libgmp-dev \
    libmpdec-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python version
ENV PYTHON_VERSION=3.11.9

# Download and extract Python source
RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz \
    && tar -xf Python-${PYTHON_VERSION}.tgz \
    && rm Python-${PYTHON_VERSION}.tgz

# Compile and install Python
RUN cd Python-${PYTHON_VERSION} \
    && ./configure --enable-optimizations \
    && make -j "$(nproc)" \
    && make altinstall

# Remove the source directory to clean up
RUN rm -rf Python-${PYTHON_VERSION}

# Create a symlink to make 'python3' point to the new Python version
RUN ln -s /usr/local/bin/python3.11 /usr/bin/python3

# **New Step: Create a symlink for 'python' to point to 'python3'**
RUN ln -s /usr/bin/python3 /usr/bin/python
# 升级 pip 到最新版本
RUN python3 -m pip install --upgrade pip

# 创建 'pip3' 的符号链接指向新安装的 pip3.11
RUN ln -s /usr/local/bin/pip3.11 /usr/local/bin/pip3

# 创建 'pip' 的符号链接指向 'pip3'
RUN ln -s /usr/local/bin/pip3 /usr/local/bin/pip
# 设置非交互模式，避免 tzdata 等软件包配置交互提示
ENV DEBIAN_FRONTEND=noninteractive


# 设置工作目录
WORKDIR /code

# 复制项目文件到镜像中
COPY . .

# 创建启动脚本
RUN chmod +x /code/start.sh

# 启动命令：执行启动脚本
#CMD ["/code/start.sh"]
