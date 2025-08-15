import os
from slurm_jobs.utils import setup


def get_train_cmd(noise_config, corruption_probability, dataset, dataset_path, checkpoint=None, dp=1.0, duration=None, weight_decay=0.0, all_pairs=False, overwrite_cls_labels_path=None, crop_size=None, precond="edmcls"):
    outdir = os.environ["BASE_PATH"]

    imagenet_params = f"--batch=4096 --dump=200 --cond=1 --arch=adm --lr=1e-4 --dropout=0.1 --augment=0.0 --fp16=1 --ls=100 --duration={2500 if duration is None else duration} --ema=50 "
    cifar_params = f"--cond=0 --arch=ddpmpp --dump=50 --duration={200 if duration is None else duration} --lr=1e-4"
    celeba_params = f"--batch=512 --dump=200 --cond=0 --arch=ddpmpp --cres=1,2,2,2 --lr=2e-4 --dropout=0.1 --augment=0.15 --duration={100 if duration is None else duration} "
    params = imagenet_params if "imagenet" in dataset.lower() else cifar_params if "cifar" in dataset.lower() else celeba_params


    
    if overwrite_cls_labels_path is not None:
        overwrite_cls_labels_path = os.path.abspath(overwrite_cls_labels_path)
        overwrite_cls_labels_path = f"--overwrite_cls_labels_path={overwrite_cls_labels_path}"
    else:
        overwrite_cls_labels_path = ""
    
    if crop_size is not None:
        crop_size = f"--crop_size={crop_size}"
    else:
        crop_size = ""

    cmd = f"""ibrun python -m torch.distributed.run \
            --nnodes=$SLURM_JOB_NUM_NODES --rdzv_id=$RANDOM --rdzv_backend=c10d --rdzv_endpoint $head_node_ip:29500 --nproc_per_node=1 train.py \
            --outdir={outdir} \
            --data={dataset_path} \
            {params} \
            --precond={precond} \
            --noise_config={noise_config} \
            --corruption_probability={corruption_probability} \
            --dataset_keep_percentage={dp} \
            --weight_decay={weight_decay} \
            --all_pairs={all_pairs} \
            {overwrite_cls_labels_path} \
            {crop_size} \
            --expr_id=train_cls_{noise_config}_{dataset.lower()}_dp{dp}_wd{weight_decay}_all_pairs{all_pairs}"""
    if checkpoint is not None:
        cmd += f" --resume={checkpoint}"
    return cmd

def run():
    dataset = os.environ["DATASET"]
    dataset_path = os.environ["DATASET_PATH"]
    max_index = os.environ.get("MAX_INDEX", None)
    crop_size = os.environ.get("CROP_SIZE", None)
    precond = os.environ.get("PRECOND", "edmcls")
    if max_index is None:
        # set based on dataset
        max_index = 200_000 if dataset == "cifar" else 100_000 if dataset == "celeba" else 2_500_000 if dataset == "imagenet" else None
    else:
        max_index = int(max_index)
    
    noise_config = os.environ["NOISE_CONFIG"] if "NOISE_CONFIG" in os.environ else "blurs"
    corruption_probability = float(os.environ["CORRUPTION_PROBABILITY"])
    dp = float(os.environ["DP"]) if "DP" in os.environ else 1.0
    dry_run = os.environ.get("DRY_RUN", False)
    duration = float(os.environ["DURATION"]) if "DURATION" in os.environ else None
    weight_decay = float(os.environ["WEIGHT_DECAY"]) if "WEIGHT_DECAY" in os.environ else 0.0
    all_pairs = os.environ.get("ALL_PAIRS", False)
    overwrite_cls_labels_path = os.environ.get("OVERWRITE_CLS_LABELS_PATH", None)
    cmd = "echo Running \n"

        
    if not dry_run:
        # fix resuming from checkpoint when the training instability is addressed.
        cmd += get_train_cmd(noise_config, corruption_probability, dataset, dataset_path, dp=dp, duration=duration, weight_decay=weight_decay, all_pairs=all_pairs, overwrite_cls_labels_path=overwrite_cls_labels_path, crop_size=crop_size, precond=precond)
    else:
        cmd += "echo 'Dry run'"
    print(cmd)

if __name__ == "__main__":
    run()



