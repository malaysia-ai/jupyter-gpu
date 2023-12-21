FROM rocm/pytorch:rocm5.7_ubuntu22.04_py3.10_pytorch_2.0.1

# RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1001 ubuntudocker
RUN apt update
RUN apt install python3 python3-dev python3-pip -y
RUN apt install sudo -y
RUN adduser --quiet --disabled-password --shell /bin/bash --home /home/ubuntu --gecos "User" ubuntu
RUN usermod -aG sudo ubuntu
RUN echo "ubuntu ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER ubuntu
WORKDIR /home/ubuntu

RUN sudo apt install curl -y
RUN sudo apt install git telnet cron vim -y
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
RUN sudo apt install nodejs -y
RUN sudo npm install pm2 -g
RUN node --version
RUN npm --version
RUN pm2 list

# https://gist.github.com/cgmb/6ae0d118bf357fc4576a7568b85e1c45
# https://www.reddit.com/r/ROCm/comments/11j1cvs/cloud_vms_with_amd_gpus_having_hip_support/
RUN sudo apt-get update && \
    sudo apt-get -y upgrade && \
    sudo apt-get -y install linux-modules-extra-aws

RUN sudo apt-get update && \
    sudo apt-get install -y cmake build-essential g++-12
COPY . /opt/cmakeamd

RUN sudo apt-get update -y
RUN sudo apt-get install kmod

# jupyter
RUN curl -fsSL https://code-server.dev/install.sh | sh
RUN sudo mkdir /etc/jupyter
COPY jupyter_notebook_config.py /etc/jupyter
RUN sudo pip3 install jupyter notebook==6.5.6 jupyter-server==1.18.0 jupyter-server-proxy==3.2.1

RUN sudo echo "export PATH=\"$HOME/.local/bin:$PATH\"" > .bashrc

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
RUN sudo apt-get install git-lfs

RUN sudo apt-get update \
&& sudo apt-get install -y wget gnupg \
&& wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | sudo apt-key add - \
&& sudo sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list' \
&& sudo apt-get update \
&& sudo apt-get install -y google-chrome-stable fonts-ipafont-gothic fonts-wqy-zenhei fonts-thai-tlwg fonts-kacst fonts-freefont-ttf libxss1 \
    --no-install-recommends

RUN sudo ssh-keygen -A
RUN sudo apt install openssh-server sshpass -y
RUN echo ubuntu:ubuntu123 | sudo chpasswd
