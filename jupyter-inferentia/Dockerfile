FROM ubuntu:20.04

# RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1001 ubuntu
RUN apt update

ARG VERSION_CODENAME=bionic

RUN echo "deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main" > /etc/apt/sources.list.d/neuron.list

RUN apt install wget gnupg2 -y

RUN wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | apt-key add -

# Update OS packages 
RUN apt-get update -y

# Install git 
RUN apt-get install git -y

RUN apt-get install aws-neuronx-dkms=2.* -y
RUN apt-get install aws-neuronx-collectives=2.* -y
RUN apt-get install aws-neuronx-runtime-lib=2.* -y
RUN apt-get install aws-neuronx-tools=2.* -y

RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata

# Add PATH
ENV PATH="/opt/aws/neuron/bin:$PATH"

RUN apt install python3 python3-dev python3-pip -y
RUN apt install sudo -y
RUN adduser --quiet --disabled-password --shell /bin/bash --home /home/ubuntu --gecos "User" ubuntu
RUN usermod -aG sudo ubuntu
RUN echo "ubuntu ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER ubuntu
WORKDIR /home/ubuntu

RUN sudo apt install curl -y
RUN sudo apt install telnet cron vim -y
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
RUN sudo apt install nodejs -y
RUN sudo npm install pm2 -g
RUN node --version
RUN npm --version
RUN pm2 list

RUN curl -fsSL https://code-server.dev/install.sh | sh
RUN sudo mkdir /etc/jupyter
COPY jupyter_notebook_config.py /etc/jupyter
RUN sudo pip3 install jupyter notebook==6.4.12 jupyter-server==1.18.0 jupyter-server-proxy==3.2.1

RUN echo "export PATH=\"$HOME/.local/bin:$PATH\"" > .bashrc

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

RUN sudo python3 -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

RUN sudo python3 -m pip install neuronx-cc==2.* --pre torch-neuronx==2.0.* torchvision

RUN sudo pip3 install traitlets==5.9.0