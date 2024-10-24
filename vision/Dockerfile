FROM uhub.service.ucloud.cn/oldwang12/python:3.8.19
WORKDIR /v_app
COPY . /v_app
RUN apt-get update && apt-get install -y dpkg
COPY MVS-3.0.1_x86_64_20240902.deb /tmp/
RUN dpkg -i /tmp/MVS-3.0.1_x86_64_20240902.deb
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
RUN rm /tmp/MVS-3.0.1_x86_64_20240902.deb
RUN apt-get clean
EXPOSE 8000
CMD ["python", "Main.py"]
