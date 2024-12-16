# Use the official Ubuntu base image
FROM ubuntu:22.04

# Set environment variables to non-interactive to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update  \
    && apt install wget  libgl1 libglib2.0-0 -y



 # Activate the virtual environment and install necessary Python packages
RUN wget -q https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda-latest-Linux-x86_64.sh -O Miniconda3.sh && \
    bash Miniconda3.sh -b -p /opt/miniconda && \
    rm Miniconda3.sh && \
    /opt/miniconda/bin/conda create --yes --name  MinerU python=3.10

RUN /bin/bash -c "source /opt/miniconda/etc/profile.d/conda.sh && \
    conda activate MinerU && \
    pip3 install --upgrade pip && \
    pip install -U magic-pdf[full] --extra-index-url https://wheels.myhloli.com -i https://mirrors.aliyun.com/pypi/simple --retries=300"

RUN /bin/bash -c "source /opt/miniconda/etc/profile.d/conda.sh && \
    conda activate MinerU && \
    pip install modelscope  -i https://mirrors.aliyun.com/pypi/simple  --retries=300 && \
    wget https://gitee.com/myhloli/MinerU/raw/master/scripts/download_models.py -O download_models.py && \
    python download_models.py"

ENTRYPOINT ["/bin/bash", "-c", "source /opt/miniconda/etc/profile.d/conda.sh && conda activate MinerU && exec \"$@\"", "--"]