sudo apt update
sudo apt-get install -y ca-certificates curl gnupg
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg
NODE_MAJOR=18
echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_$NODE_MAJOR.x nodistro main" | sudo tee /etc/apt/sources.list.d/nodesource.list
sudo apt-get update
sudo apt-get install nodejs -y
sudo npm install -g configurable-http-proxy
sudo apt install python3-dev python3-pip -y
sudo pip3 install notebook jupyterlab jupyterhub jupyter-server-proxy
sudo mkdir -p /etc/jupyter
sudo mkdir -p /etc/jupyterhub
sudo cp jupyter_notebook_config.py /etc/jupyter
sudo cp jupyterhub_config.py /etc/jupyterhub
curl -fsSL https://code-server.dev/install.sh | sh

sudo cp jupyterhub.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable jupyterhub.service 
sudo systemctl start jupyterhub.service