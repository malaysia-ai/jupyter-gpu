# Building jupyter image from Neuronx

1. Authenticate Docker with neuronx ECR Registry [here](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#neuron-containers)

```bash
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-west-2.amazonaws.com
docker pull 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-neuronx:1.13.1-neuronx-py310-sdk2.15.0-ubuntu20.04
```

2. Extend [torch-neuronx](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#neuron-containers) image to include jupyter in dockerfile

```dockerfile
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-neuronx:1.13.1-neuronx-py310-sdk2.15.0-ubuntu20.04
.
.
.
```

3. push image into dockerhub 

```bash
docker build --progress=plain -t malaysiaai/jupyter-inferentia-neuron:main .
docker push malaysiaai/jupyter-inferentia-neuron:main
```
