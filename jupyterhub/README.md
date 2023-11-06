# Jupyterhub GPU

## Building image

```bash
docker build -t malaysiaai/jupyterhub-gpu-devel:main .
docker push malaysiaai/jupyterhub-gpu-devel:main
```

## how-to install

1. Install Jupyterhub using Helm,

```bash
helm repo add jupyterhub https://jupyterhub.github.io/helm-chart/
helm upgrade --cleanup-on-fail \
--install jupyterhub jupyterhub/jupyterhub \
--namespace jupyterhub \
--create-namespace \
--values config.yaml
```

Full config at https://github.com/jupyterhub/zero-to-jupyterhub-k8s/blob/main/jupyterhub/values.yaml