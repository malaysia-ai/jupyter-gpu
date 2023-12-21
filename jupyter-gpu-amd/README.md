
1. The image [malaysiaai/jupyter-gpu-amd:latest](https://hub.docker.com/repository/docker/malaysiaai/jupyter-gpu-amd/general) is built upon the rocm image, as detailed in AMD's ROCm Documentation for Docker Image Support Matrix [here](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/docker-image-support-matrix.html).

2. We pushed the image into our dockerhub repository
```
docker build malaysiaai/jupyter-gpu-amd:v1
docker tag malaysiaai/jupyter-gpu-amd:v1 malaysiaai/jupyter-gpu-amd:v1
docker push malaysiaai/jupyter-gpu-amd:v1
```
