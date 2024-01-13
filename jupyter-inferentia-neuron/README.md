# Building jupyter on Neuronx image
1. Authenticate Docker with neuronx ECR Registry [here](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#user-content-neuron-containers).

```bash
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-west-2.amazonaws.com

# testing whether pulling works or not
docker pull 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-neuronx:1.13.1-neuronx-py310-sdk2.15.0-ubuntu20.04
```

2. Extend torch-neuronx image to include jupyter in dockerfile (image [here](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#user-content-neuron-containers)).


```dockerfile
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-neuronx:1.13.1-neuronx-py310-sdk2.15.0-ubuntu20.04
.
.
.
```

3. build and push image into malaysia-ai dockerhub.

```bash
docker build --progress=plain -t malaysiaai/jupyter-inferentia-neuron:main .
docker push malaysiaai/jupyter-inferentia-neuron:main
```
