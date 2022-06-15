# Jupyterhub + Code-server

Jupyterhub with GPU and code-server! If you part of the organization, you can access to jupyterhub! https://jupyterhub.malaysiaai.ml

```text
Thu Jun  2 01:07:23 2022       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.129.06   Driver Version: 470.129.06   CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  Off |
| 77%   67C    P2   407W / 450W |  13690MiB / 24247MiB |    100%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      1619      G   /usr/lib/xorg/Xorg                 35MiB |
|    0   N/A  N/A      3276      G   /usr/lib/xorg/Xorg                 95MiB |
|    0   N/A  N/A    113211      G   /usr/bin/gnome-shell               37MiB |
|    0   N/A  N/A    329691      G   /usr/lib/firefox/firefox           13MiB |
|    0   N/A  N/A    450819      C   python3                         13481MiB |
+-----------------------------------------------------------------------------+
```

## Rules

1. Notebooks automatically shutdown if idle more than 120 minutes.
2. Admin can kill any GPU usage app anytime.
3. Admin also can access your directory anytime.

## how-to

1. Install JupyterHub and Code-server,

```bash
sudo apt update
sudo apt-get install nodejs npm
sudo apt install python3-dev python3-pip -y
sudo pip3 install notebook jupyterlab jupyterhub jupyter-server-proxy
sudo mkdir -p /etc/jupyter
sudo cp jupyter_notebook_config.py /etc/jupyter
sudo cp jupyterhub_config.py /etc/jupyter
curl -fsSL https://code-server.dev/install.sh | sh

sudo cp jupyterhub.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable jupyterhub.service 
sudo systemctl start jupyterhub.service
```

## Access Jupyterlab

Just go to /user/{username}/lab

<img alt="logo" width="40%" src="jupyterlab.png">

## Access code-server

Just go to /user/{username}/code-server

<img alt="logo" width="40%" src="code-server.png">

## new Kernel

Let say you want to have your own virtualenv want to deploy it as jupyter kernel, follow simple step below,

1. Initialize virtual env,

```bash
python3 -m venv tf-nvidia
```

2. Add in jupyter notebook kernel,

```bash
~/tf-nvidia/bin/pip3 install ipykernel
~/tf-nvidia/bin/python3 -m ipykernel install --user --name=tf1
```

You will found your new kernel in Jupyter Notebook as `tf1`.
