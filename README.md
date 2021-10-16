# Jupyterhub
Jupyterhub with GPU. If you part of the organization, you can access to jupyterhub! https://jupyterhub.malaysiaai.ml

```text
Sat Oct  2 21:19:00 2021       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.91.03    Driver Version: 460.91.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  GeForce RTX 3080    Off  | 00000000:01:00.0 Off |                  N/A |
|  0%   39C    P8    10W / 320W |     29MiB / 10018MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A       880      G   /usr/lib/xorg/Xorg                 15MiB |
|    0   N/A  N/A      1161      G   /usr/bin/gnome-shell                8MiB |
|    0   N/A  N/A      1456      G   ...bexec/gnome-initial-setup        3MiB |
+-----------------------------------------------------------------------------+
```

## Rules

1. Directory home in Jupyter Notebook is shared among users.
2. Make sure naming your directory properly.
3. Do not try to delete other users data.
4. Notebooks automatically shutdown if idle more than 5 minutes.
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
