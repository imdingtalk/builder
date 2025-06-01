FROM continuumio/miniconda3:4.12.0

# ADD sources.list /etc/apt/
# 基础环境更新
RUN apt-get update -y && apt-get install -y gcc vim && apt-get clean && rm -rf /var/lib/apt/lists/* 

# ADD .condarc /root/
# mamba加速
RUN conda install -y -c conda-forge mamba==0.23.1 \
    # 创建py38环境
    && mamba create -n py38 -y python==3.8.13 \
    # pytorch
    # && mamba install -n py38 -y -c pytorch pytorch==2.0.0 torchaudio==2.0.0 torchvision==0.15.0 cpuonly==2.0 \
    # 安装气象依赖
    && mamba install -n py38 -y -c conda-forge metpy==1.4.0 pynio==1.5.5 satpy==0.36.0 matplotlib==3.7.0 \
    pygrib==2.1.4 xesmf==0.8.2 regionmask==0.9.0 geographiclib==2.0 numpy==1.23.5 numba==0.56.4 scipy==1.9.1 wrf-python==1.3.4.1 \
    xarray==2022.11.0 geopandas==0.12.2 scikit-image==0.20.0 scikit-learn==1.2.2 xgboost==2.0.3 \
    # 安装通用依赖
    openpyxl==3.0.10 json5==0.9.5 boto3==1.26.9 docxtpl==0.11.5 paramiko==3.1.0 pyjwt==2.4.0 pdfplumber==0.9.0 python-multipart==0.0.7 \
    # 数据库
    sqlalchemy==1.4.44 psycopg2==2.9.3 pymysql==1.0.2 oracledb==1.3.1 presto-python-client==0.8.3 \
    # 清理
    && conda uninstall -y -c conda-forge mamba==0.23.1 && conda clean -a -y
# 无法用conda的一些包
RUN /opt/conda/envs/py38/bin/pip install cython==0.29.32 --no-cache-dir
# -i https://pypi.mirrors.ustc.edu.cn/simple/
# RUN /opt/conda/envs/py38/bin/pip install cinrad==1.7.1 pycwr==0.3.5 opencv-python-headless==4.7.0.72 meteva==1.7.5 --no-cache-dir -i https://pypi.mirrors.ustc.edu.cn/simple/

# 安装prefect依赖
RUN /opt/conda/envs/py38/bin/pip install prefect==2.8.6 pydantic==1.10.7 apprise==1.6.0 anyio==3.7.1 pendulum==2.1.2 importlib_metadata==6.8.0 --no-cache-dir
# -i https://pypi.mirrors.ustc.edu.cn/simple/

# 临时添加
RUN /opt/conda/envs/py38/bin/pip install statsmodels==0.14.1 --no-cache-dir
# -i https://pypi.mirrors.ustc.edu.cn/simple/

# 添加工具包
# ADD cli /cli

# 启动相关
ENV TZ=Asia/Shanghai
ENV PY_PATH=/opt/conda/envs/py38
ENV ESMFMKFILE=/opt/conda/envs/py38/lib/esmf.mk
ENV PROJ_LIB=/opt/conda/envs/py38/share/proj
WORKDIR /code
RUN echo '/bin/bash startAgent.sh' > /entrypoint.sh && chmod +x /entrypoint.sh
CMD [ "/bin/bash", "-c", "/entrypoint.sh"]
