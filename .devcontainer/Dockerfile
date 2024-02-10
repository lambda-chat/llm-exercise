FROM debian:bookworm

RUN apt-get update \
    ## [optional] to install fish 
    && apt-get install -y python3-launchpadlib software-properties-common \
    && apt-add-repository -y ppa:fish-shell/release-3 \
    && apt-get update \
    ## install essential tools
    && apt-get install -y \
    # optional
    fish \
    # essential
    # libgl1-mesa-glx libglib2.0-0 \
    # for remote user
    sudo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# add a non-root user `vscode` with sudo access
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

CMD ["/bin/sh", "-c", "sleep infinity"]
