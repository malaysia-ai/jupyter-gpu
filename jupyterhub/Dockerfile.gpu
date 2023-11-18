FROM nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu22.04

RUN apt update
RUN apt install python3 python3-dev python3-pip -y
RUN apt install sudo -y
RUN adduser --quiet --disabled-password --shell /bin/bash --home /home/ubuntu --gecos "User" ubuntu
RUN usermod -aG sudo ubuntu
RUN echo "ubuntu ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN apt-get install lsof telnet htop graphviz unzip ssh-client openssh-server -y

RUN pip3 install jupyterlab notebook jupyterhub nbclassic

RUN jupyter server --generate-config && jupyter lab clean && fix-permissions /home/ubuntu

RUN curl -fsSL https://code-server.dev/install.sh | sh

COPY jupyter_notebook_config.py /etc/jupyter

RUN pip3 install jupyter-server-proxy

RUN pip3 install jupyter_contrib_nbextensions autopep8

ENV JUPYTER_PORT=8888
EXPOSE $JUPYTER_PORT

# Configure container startup
CMD ["start-notebook.py"]

# Copy local files as late as possible to avoid cache busting
COPY start-notebook.py start-notebook.sh start-singleuser.py start-singleuser.sh /usr/local/bin/
COPY jupyter_server_config.py docker_healthcheck.py /etc/jupyter/

# Fix permissions on /etc/jupyter as root
USER root
RUN fix-permissions /etc/jupyter/

# HEALTHCHECK documentation: https://docs.docker.com/engine/reference/builder/#healthcheck
# This healtcheck works well for `lab`, `notebook`, `nbclassic`, `server` and `retro` jupyter commands
# https://github.com/jupyter/docker-stacks/issues/915#issuecomment-1068528799
HEALTHCHECK --interval=5s --timeout=3s --start-period=5s --retries=3 \
    CMD /etc/jupyter/docker_healthcheck.py || exit 1

USER ubuntu

WORKDIR /home/ubuntu

RUN jupyter contrib nbextension install --user

RUN jupyter nbextension enable varInspector/main