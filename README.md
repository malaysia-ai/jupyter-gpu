# Jupyterhub
Jupyterhub with GPU. If you part of the organization, you can access to jupyterhub! https://jupyterhub.malaysiaai.ml

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

1. Directory home in Jupyter Notebook is shared among users.
2. Make sure naming your directory properly.
3. Do not try to delete other users data.
4. Notebooks automatically shutdown if idle more than 10 minutes.
5. Admin can kill any GPU usage app anytime.

## how-to

1. Install Jupyter Notebook,

```bash
sudo apt update
sudo apt install python3-dev python3-pip -y
pip3 install jupyter

sudo cp jupyterhub.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable jupyterhub.service 
sudo systemctl start jupyterhub.service
```

## new kernel

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
