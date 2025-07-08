FROM debian:11.5

ENV container=docker \
    PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=America/New_York \
    HOME=/home/trader \
    PYENV_ROOT=/home/trader/.pyenv \
    TMPDIR=/home/trader/.tmp

ENV PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$HOME/.local/bin:$PATH"

RUN useradd -m -d /home/trader -s /bin/bash -G sudo trader && \
    mkdir -p /var/run/sshd /run/sshd /tmp

RUN apt-get update && apt-get install -y gnupg gnupg2 \
    dialog apt-utils \
    python3 python3-pip python3-venv \
    git wget vim dpkg curl unzip \
    build-essential \
    tzdata \
    linux-headers-generic lzip \
    locales-all \
    redis-server \
    openssh-server \
    sudo \
    tmux expect \
    iproute2 net-tools rsync iputils-ping \
    lnav jq \
    libgtk-3-0  libasound2 libnss3 libgbm1 libnspr4 \
    libbz2-dev libsqlite3-dev libreadline-dev \
    libedit-dev libncurses5-dev libssl-dev libffi-dev liblzma-dev \
    libtiff5-dev libtiff5 libjbig-dev \
    libpango-1.0-0 libpangocairo-1.0-0 libcairo2-dev libgdk-pixbuf2.0-0 \
    tigervnc-scraping-server xvfb xterm && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN wget -qO - https://www.mongodb.org/static/pgp/server-5.0.asc | apt-key add - && \
    echo "deb http://repo.mongodb.org/apt/debian bullseye/mongodb-org/5.0 main" | tee /etc/apt/sources.list.d/mongodb-org-5.0.list && \
    apt-get update && apt-get install -y mongodb-org && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN echo 'trader:trader' | chpasswd

RUN mkdir -p \
    /home/trader/ibc/logs \
    /home/trader/mmr/data/redis \
    /home/trader/mmr/data/mongodb \
    /home/trader/mmr/logs \
    /home/trader/.vnc \
    /home/trader/.config \
    /home/trader/.tmp \
    /home/trader/.cache

RUN echo 'trader' | vncpasswd -f > /home/trader/.vnc/passwd

COPY ./ /home/trader/mmr/

RUN wget https://github.com/IbcAlpha/IBC/releases/download/3.16.2/IBCLinux-3.16.2.zip -P /home/trader/mmr/third_party && \
    unzip -o /home/trader/mmr/third_party/IBCLinux-3.16.2.zip -d /home/trader/mmr/third_party/ibc && \
    rm -f /home/trader/mmr/third_party/IBCLinux-3.16.2.zip && \
    chmod +x /home/trader/mmr/third_party/ibc/*.sh

COPY ./scripts/installation/.bash_profile /home/trader/
RUN touch /home/trader/.hushlogin \
    /home/trader/mmr/logs/trader_service.log \
    /home/trader/mmr/logs/strategy_service.log

RUN chown -R trader:trader /home/trader

USER trader
WORKDIR /home/trader/mmr

RUN curl https://pyenv.run | bash && \
    pyenv install 3.10.11 && \
    pyenv virtualenv 3.10.11 mmr && \
    pyenv rehash

RUN curl -sSL https://install.python-poetry.org | python3 -

RUN --mount=type=cache,target=/home/trader/.cache/pip \
    pip3 install -r /home/trader/mmr/requirements.txt

EXPOSE 22 7496 6379 27017 8081 5900 5901

USER root
WORKDIR /home/trader

ENTRYPOINT ["sh", "-c", "service ssh restart && tail -f /dev/null"]
