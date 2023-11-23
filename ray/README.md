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

## Dry run

To dry run,

```bash
python3 train.py \
--model_name_or_path huseinzol05/dummy-mistral-1.1b \
--per_device_train_batch_size 24 \
--gradient_accumulation_steps 1 \
--storage_directory "/home/ubuntu" \
--output_dir mistral-1.1b \
--bf16 \
--torch_dtype "bfloat16" \
--do_train \
--do_eval false \
--num_train_epochs 2 \
--train_file "s3://train/indexed" \
--logging_steps 1 \
--learning_rate 2e-4 \
--block_size 4096 \
--save_steps 10 \
--save_total_limit 3 \
--warmup_steps 50 \
--gradient_checkpointing true \
--num_workers 8
```

```
 23%|██▎       | 5/22 [00:42<02:21,  8.35s/it]
(RayTrainWorker pid=20282, ip=10.224.0.177) {'loss': 8.4434, 'learning_rate': 8.228161798644421e-05, 'epoch': 0.45}
 23%|██▎       | 5/22 [00:43<02:24,  8.48s/it]
(RayTrainWorker pid=20282, ip=10.224.0.177) {'loss': 8.3139, 'learning_rate': 9.160270615698787e-05, 'epoch': 0.55}
 27%|██▋       | 6/22 [00:51<02:12,  8.28s/it]
 27%|██▋       | 6/22 [00:50<02:11,  8.19s/it]
(RayTrainWorker pid=56577) {'loss': 8.3139, 'learning_rate': 9.160270615698787e-05, 'epoch': 0.55}
(RayTrainWorker pid=20282, ip=10.224.0.177) {'loss': 7.9451, 'learning_rate': 9.948357391330555e-05, 'epoch': 0.64}
 32%|███▏      | 7/22 [00:59<02:01,  8.09s/it]
 32%|███▏      | 7/22 [00:57<02:00,  8.03s/it]
(RayTrainWorker pid=56577) {'loss': 7.9451, 'learning_rate': 9.948357391330555e-05, 'epoch': 0.64}
```

- tested to save the checkpoints and prune old checkpoints.
- tested to load the checkpoint.

https://wandb.ai/mesolitica/run-ray?workspace=user-husein-mesolitica

[train.py](train.py) already hardcoded deepspeed Zero 3 config.

## Building image

```bash
docker build -t malaysiaai/ray-gpu-devel:main .
docker push malaysiaai/ray-gpu-devel:main
```