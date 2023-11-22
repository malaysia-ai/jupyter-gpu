# Training with Ray

The point of this README, is to prep developers to train multinodes on SPOT instances.

## Why Spot?

GPUs are expensive bro.

## Prerequisite

1. Distributed storage, if you want privacy and local network which is way faster. Go with MinIO, minimum 2 cores 8GB RAM if you want MinIO faster.

Much better if we use MinIO cluster, use bitnami chart instead.

## What do we learned?

### Where are the notebooks?

1. [train-gpt2-mosaic.ipynb](train-gpt2-mosaic.ipynb), 2 nodes, trained GPT2, each node is 1x A100 GPU.

- tested to save the checkpoints and prune old checkpoints.
- tested to load the checkpoint.
- tested random crashed on workers.
- tested random crashed on master.

2. [multigpus-multinodes.ipynb](multigpus-multinodes.ipynb), 2 nodes, each node is 4x A100 GPUs.

- tested to save the checkpoints and prune old checkpoints.
- tested to load the checkpoint.
- tested random crashed on workers.
- tested random crashed on master.

### Ray storage sucks

If you used to HuggingFace Trainer interface, loading checkpoint is very easy, I mean, it just checkpoints, but in Ray, even you setup distributed storage, loading checkpoint is straight sucks for HuggingFace, https://docs.ray.io/en/latest/train/user-guides/checkpoints.html#train-distributed-checkpointing

So to solve this problem, we created S3 Callback during HuggingFace `on_save`, so this will sync from local to S3.

During starting train session, we only fetch latest checkpoint to each locals and load as usual.

What is the downside of this? **each locals** need to fetch the same remote checkpoint, this can burden remote storage.

### HuggingFace datasets super slow on huge dataset

We are talking about 100GB+ text files, the problem with HuggingFace datasets, it stream memory mapped file and after that concat, https://github.com/huggingface/datasets/blob/60bdf3005d1dc0b26da8e5949721b20d932eaad6/src/datasets/table.py#L51, super super slow.

You might read https://github.com/huggingface/datasets/issues/2252#issuecomment-846004656, `load_from_disk` can solve the problem, but the problem with this method, it will download entire partitions into each locals.

We cannot use iterator dataset, to resume last steps is not possible due to behavior of iterator (no dataset length).

So to solve this, we use MosaicML streaming, read [test-mosaic.ipynb](test-mosaic.ipynb) how to prepare the dataset and upload to remote.

### MosaicML streaming is weird

There are locks mechanism happened on first iteration, but, by doing this,

```python
# https://github.com/mosaicml/streaming/issues/307#issuecomment-1729829065
def inf_loop_dataloader(dataloader: torch.utils.data.DataLoader):
    while True:
        for batch in dataloader:
            yield batch
dataloader = DataLoader(train_dataset, batch_size=2)
dataset_iterator = iter(inf_loop_dataloader(dataloader))
batch = next(iter(dataset_iterator))
```

Solved the problem.

### 1 GPU == 1 worker

To utilize all GPUs available, you must set worker size == number of gpus. If you have 2 nodes, each node got 4 GPUs, so the number of workers is 8.

## What if

### Worker died

If you set,

```python
run_config = train.RunConfig(failure_config=train.FailureConfig(max_failures=-1))
ray_trainer = TorchTrainer(
    train_func,
    scaling_config=scaling_config,
    run_config=run_config

)
```

Everything should be ok, the master just wait the worker come back.

### Master died

```text
RayActorError: The actor died unexpectedly before finishing this task.
	class_name: _QueueActor
	actor_id: 317b9ec747c7ca69b0d3016905000000
	pid: 12433
	namespace: 156b8ae5-a686-44c3-a310-a0d4e01f3334
	ip: 10.208.0.212
The actor is dead because its owner has died. Owner Id: 05000000ffffffffffffffffffffffffffffffffffffffffffffffff Owner Ip address: 10.208.0.212 Owner worker exit type: SYSTEM_ERROR Worker exit detail: Owner's node has crashed.
```

The script totally dead, to solve this problem, we can use context manager and infinite loop,

```python
import time

class RayConnection:
    def __init__(self, address, **kwargs):
        ray.init(address=address, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, typ, value, traceback):
        ray.shutdown()


while True:
    with RayConnection("ray://ray-master:10001", runtime_env=runtime_env):
        scaling_config = ScalingConfig(num_workers=2, use_gpu=True)
        run_config = train.RunConfig(failure_config=train.FailureConfig(max_failures=-1))
        ray_trainer = TorchTrainer(
            train_func,
            train_loop_config={
                'local': 'local_dir',
                'remote': 's3://train/indexed'
            },
            scaling_config=scaling_config,
            run_config=run_config

        )
        result = ray_trainer.fit()
    
    print('ray cluster disconnected, time to sleep.')
    time.sleep(10)
```

But, training script will be inside ray master, so if master died, script died too.