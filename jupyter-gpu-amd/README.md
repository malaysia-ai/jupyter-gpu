
The image [malaysiaai/jupyter-gpu-amd:latest](https://hub.docker.com/repository/docker/malaysiaai/jupyter-gpu-amd/general) is built upon the [rocm image](https://hub.docker.com/layers/rocm/pytorch/rocm5.7_ubuntu22.04_py3.10_pytorch_2.0.1/images/sha256-21df283b1712f3d73884b9bc4733919374344ceacb694e8fbc2c50bdd3e767ee), as detailed in AMD's ROCm Documentation for Docker Image Support Matrix below.


https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/docker-image-support-matrix.html


```
docker tag malaysiaai/jupyter-gpu-amd:latest malaysiaai/jupyter-gpu-amd:latest
docker push malaysiaai/jupyter-gpu-amd:latest
```
