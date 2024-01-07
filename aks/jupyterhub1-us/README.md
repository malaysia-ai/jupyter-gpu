# deploy jupyterhub into aks

1. Add repo chart [reference here](https://z2jh.jupyter.org/en/stable/jupyterhub/installation.html#install-jupyterhub)
```
helm repo add jupyterhub https://hub.jupyter.org/helm-chart/
helm repo update
```

2. install helm [reference here](https://z2jh.jupyter.org/en/stable/administrator/authentication.html#github)

reference: 
```
helm upgrade --cleanup-on-fail \
  --install <helm-release-name> jupyterhub/jupyterhub \
  --namespace <k8s-namespace> \
  --create-namespace \
  --version=<chart-version> \
  --values config.yaml
```

```
helm upgrade --cleanup-on-fail 
--install jupyterhub jupyterhub/jupyterhub
 --namespace jupyterhub1 
 --create-namespace 
 --values values.yaml
```

3. Ingress jupyterhub
```
kubectl apply -f ingress.yaml
```