# Jupyterhub
Jupyterhub with GPU. If you part of the organization, you can access to jupyterhub! https://jupyterhub.malaysiaai.ml

## how-to

1. Install Jupyter Notebook,

```bash
sudo apt update
sudo apt install python3.8 -y
sudo apt install python3-pip python3.8-dev -y
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
sudo apt install npm -y
sudo npm install pm2 -g
curl -fsSL https://deb.nodesource.com/setup_12.x | sudo -E bash -
sudo apt-get install -y nodejs
pip3 install jupyter
sudo pip3 install jupyterlab-topbar jupyterlab-pygments jupyterlab-system-monitor jupyter-resource-usage
pm2 start "~/.local/bin/jupyter notebook --NotebookApp.token='' --ip=0.0.0.0"
pm2 start "~/.local/bin/jupyter lab --NotebookApp.token='' --ip=0.0.0.0 --collaborative"
pm2 startup
sudo env PATH=$PATH:/usr/bin /usr/local/lib/node_modules/pm2/bin/pm2 startup systemd -u ubuntu --hp /home/ubuntu
pm2 save
```
