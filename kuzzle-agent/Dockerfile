FROM condaforge/mambaforge:4.12.0-0

# 设置国内conda镜像源（清华）
# RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ \
#     && conda config --set show_channel_urls yes

# 设置国内pip源（中国科技大学）
# RUN mkdir -p ~/.pip && echo "[global]\nindex-url = https://pypi.mirrors.ustc.edu.cn/simple/" > ~/.pip/pip.conf

# 设置清华apt镜像源并安装常用工具

 
RUN   apt-get update && \
    apt-get install -y gcc vim && \
    apt-get clean && rm -rf /var/lib/apt/lists/*



# 创建py38环境 + 安装所有依赖
RUN mamba create -n py38 -y python=3.8.13 && \
    mamba run -n py38 mamba install -y \
    # 气象科学包
    metpy==1.4.0 pynio==1.5.5 satpy==0.36.0 matplotlib==3.7.0 \
    pygrib==2.1.4 xesmf==0.8.2 regionmask==0.9.0 geographiclib==2.0 \
    numpy==1.23.5 numba==0.56.4 scipy==1.9.1 wrf-python==1.3.4.1 \
    xarray==2022.11.0 geopandas==0.12.2 scikit-image==0.20.0 \
    scikit-learn==1.2.2 xgboost==2.0.3 \
    # 通用工具
    openpyxl==3.0.10 json5==0.9.5 boto3==1.26.9 docxtpl==0.11.5 \
    paramiko==3.1.0 pyjwt==2.4.0 pdfplumber==0.9.0 python-multipart==0.0.7 \
    # 数据库支持
    sqlalchemy==1.4.44 psycopg2==2.9.3 pymysql==1.0.2 oracledb==1.3.1 presto-python-client==0.8.3

# pip 安装部分（用run -n执行）
RUN mamba run -n py38 pip install \
    cython==0.29.32 \
    prefect==2.8.6 pydantic==1.10.7 apprise==1.6.0 anyio==3.7.1 pendulum==2.1.2 importlib_metadata==6.8.0 \
    statsmodels==0.14.1 \
    --no-cache-dir

# 设置环境变量
ENV TZ=Asia/Shanghai
ENV PY_PATH=/opt/conda/envs/py38
ENV ESMFMKFILE=/opt/conda/envs/py38/lib/esmf.mk
ENV PROJ_LIB=/opt/conda/envs/py38/share/proj
WORKDIR /code

# 启动脚本（可替换为你自己的）
RUN echo '/bin/bash startAgent.sh' > /entrypoint.sh && chmod +x /entrypoint.sh
CMD [ "/bin/bash", "-c", "/entrypoint.sh" ]
