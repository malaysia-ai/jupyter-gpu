FROM jupyter/datascience-notebook:python-3.9.13

USER root

ARG PYTHON_VERSION=3.9

ARG HELM_VERSION=3.2.1
ARG KUBECTL_VERSION=1.17.5
ARG KUSTOMIZE_VERSION=v3.8.1

RUN apt-get update && apt-get install -y curl apt-transport-https ca-certificates gnupg \
  && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
  && echo "deb https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
  && apt-get update && apt-get install -y google-cloud-sdk

# Install kubectl (same version of aws esk)
RUN curl -sLO https://storage.googleapis.com/kubernetes-release/release/v${KUBECTL_VERSION}/bin/linux/amd64/kubectl && \
    mv kubectl /usr/bin/kubectl && \
    chmod +x /usr/bin/kubectl

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install

RUN apt-get install lsof telnet htop graphviz unzip ssh-client openssh-server -y

RUN curl -fsSL https://code-server.dev/install.sh | sh

COPY jupyter_notebook_config.py /etc/jupyter

RUN pip3 install jupyter-server-proxy

RUN pip3 install jupyter_contrib_nbextensions autopep8

RUN jupyter contrib nbextension install --user

RUN jupyter nbextension enable varInspector/main