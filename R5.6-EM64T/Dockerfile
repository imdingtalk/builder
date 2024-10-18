# 使用 Ubuntu 22.04 作为基础镜像
FROM ubuntu:22.04

# 设置工作目录
WORKDIR /public/home/96069/software

# 更新包列表并安装所需的包
RUN apt update && apt install -y wget git libomp-dev

# 下载并解压 DMU 软件包
RUN wget https://dmu.ghpc.au.dk/dmu/DMU/Linux/Current/dmuv6-R5.6-EM64T-build-2024-03-06.tar.gz \
    && tar -xzf dmuv6-R5.6-EM64T-build-2024-03-06.tar.gz \
    && rm dmuv6-R5.6-EM64T-build-2024-03-06.tar.gz

# 复制当前目录下的所有内容到容器的工作目录
COPY . .


# 设定程序的默认入口点（如果需要）
# ENTRYPOINT ["/public/home/96069/software/dmuv6/R5.6-EM64T/bin/dmu1"]

# 如果希望使用不同的命令来启动容器，可以使用 CMD 指令
CMD ["sleep", "3600"]
