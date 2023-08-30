# PostgreSQL

sandboxing PostgreSQL.

## how-to

1. Install / Upgrade helm chart,

```bash
helm repo add postgresql https://charts.bitnami.com/bitnami
helm upgrade general-postgresql postgresql/postgresql -f override-values.yaml \
--install \
--namespace=default \
--create-namespace
```