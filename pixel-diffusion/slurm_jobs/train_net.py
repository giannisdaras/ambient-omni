import os
from slurm_jobs.utils import setup


def get_train_cmd(dataset_path, checkpoint=None, dp=1.0, duration=None, weight_decay=0.0, keep_schedule=True):
    outdir = os.environ["BASE_PATH"]

    imagenet_params = f"--batch=8192 --dump=200 --cond=1 --arch=adm --lr=1e-4 --dropout=0.1 --augment=0.0 --fp16=1 --ls=100 --duration={2500 if duration is None else duration} --ema=50 "
    cifar_params = f"--cond=0 --arch=ddpmpp --dump=50 --duration={200 if duration is None else duration}"
    celeba_params = f"--batch=512 --dump=200 --cond=0 --arch=ddpmpp --cres=1,2,2,2 --lr=2e-4 --dropout=0.1 --augment=0.15 --duration={100 if duration is None else duration} "
    ffhq_params = f"--batch=512 --dump=200 --cond=0 --arch=ddpmpp --cres=1,2,2,2 --lr=2e-4 --dropout=0.1 --augment=0.15 --duration={100 if duration is None else duration} "
    afhq_params = f"--batch=512 --dump=200 --cond=0 --arch=ddpmpp --cres=1,2,2,2 --lr=2e-4 --dropout=0.1 --augment=0.15 --duration={100 if duration is None else duration} "
    
    params = imagenet_params if "imagenet" in dataset_path.lower() else cifar_params if "cifar" in dataset_path.lower() else celeba_params if "celeba" in dataset_path.lower() else ffhq_params if "ffhq" in dataset_path.lower() else afhq_params if "afhq" in dataset_path.lower() else ""

    dataset_name = dataset_path.split("/")[-1]

    # we set the corruption probability to 0.0 because here we assume that the dataset is already corrupted during the annotation process.
    cmd = f"""ibrun python -m torch.distributed.run \
            --nnodes=$SLURM_JOB_NUM_NODES --rdzv_id=$RANDOM --rdzv_backend=c10d --rdzv_endpoint $head_node_ip:29500 --nproc_per_node=1 train.py \
            --outdir={outdir} \
            --data={dataset_path} \
            {params} \
            --precond=edm \
            --corruption_probability=0.0 \
            --dataset_keep_percentage={dp} \
            --weight_decay={weight_decay} \
            --expr_id=train_net_{dataset_name}_dp{dp}_wd{weight_decay} \
            --optimizer_name=adam --keep_schedule={keep_schedule}
            """
    if checkpoint is not None:
        cmd += f" --resume={checkpoint}"
    return cmd

def run():
    dataset_path = os.environ["DATASET_PATH"]    
    dp = float(os.environ["DP"]) if "DP" in os.environ else 1.0
    dry_run = os.environ.get("DRY_RUN", False)
    duration = float(os.environ["DURATION"]) if "DURATION" in os.environ else None
    weight_decay = float(os.environ["WEIGHT_DECAY"]) if "WEIGHT_DECAY" in os.environ else 0.0
    checkpoint = os.environ.get("CHECKPOINT", None)
    keep_schedule = os.environ.get("KEEP_SCHEDULE", "True") == "True"

    cmd = "echo Running \n"

        
    if not dry_run:
        # fix resuming from checkpoint when the training instability is addressed.
        cmd += get_train_cmd(dataset_path, checkpoint, dp, duration, weight_decay, keep_schedule)
    else:
        cmd += "echo 'Dry run'"
    print(cmd)

if __name__ == "__main__":
    run()



