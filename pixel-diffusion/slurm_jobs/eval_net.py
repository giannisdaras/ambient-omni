import os
from slurm_jobs.utils import (
    setup,
    find_training_folders_based_on_params,
    find_latest_checkpoint,
    find_nearest_checkpoint,
)

def get_eval_cmd(seed_id, checkpoint, dataset, 
                 dataset_path=None,
                 dp=1.0, output_file=None, 
                 rerun=False, run_memorization=False, num_generate=50000):
    
    start_seed = int(seed_id) * num_generate
    end_seed = start_seed + num_generate - 1
    seeds = f"{start_seed}-{end_seed}"
    
    if output_file is None:
        output_file = dataset.upper() + "_EVALS.txt"

    if dataset_path is None:
        dataset_path = f"/{dataset}"
    
    # check number of files in out dir
    checkpoint_index = checkpoint.split('/')[-1].split('-')[-1].split('.')[0]
    run_id = checkpoint.split('/')[-2].split('-')[-1]
    outdir = f"eval_{dataset}_{run_id}_checkpoint{checkpoint_index}_{dataset_path.split('/')[-1]}_seed{seed_id}_dp{dp}"
    outdir = os.path.join(os.environ["EVALS_DIR"], outdir)
    num_found_files = 0
    cmd = ""
    if os.path.exists(outdir):
        num_found_files = len([name for name in os.listdir(outdir) if os.path.isfile(os.path.join(outdir, name))])

    if rerun and os.path.exists(outdir):
        cmd += f"echo 'Rerun is set. Deleting all files in {outdir}...' \n"
        cmd += f"rm -rf {outdir}/* \n"
    
    if "imagenet" in dataset.lower():
        sampling_params = " --steps=256 --S_churn=40 --S_min=0.05 --S_max=50 --S_noise=1.003"
    elif "ffhq" in dataset.lower():
        sampling_params = " --steps=40"
    elif "afhq" in dataset.lower():
        sampling_params = " --steps=40"
    else:
        sampling_params = " --steps=18"

    # check that we indeed need to run this
    if (num_found_files <= num_generate) or rerun:
        cmd += f"echo 'Running generation for {num_generate} images...' \n"
        cmd += f"""ibrun python -m torch.distributed.run --nnodes=$SLURM_JOB_NUM_NODES --rdzv_id=$RANDOM --rdzv_backend=c10d --rdzv_endpoint $head_node_ip:29500 --nproc_per_node=1 generate.py \
        --seeds={seeds} --network={checkpoint} --outdir={outdir} {sampling_params}"""
    else:
        cmd += f"echo 'Skipping generation as {num_generate} images have already been generated in {outdir}'"
    
    cmd += "\n echo 'Images generated...' \n"
    
    if run_memorization:
        cmd += "\n echo 'Running memorization...'"
        # create memorization folder in the same root folder as
        memorization_dir = outdir + "_memorization"
        os.makedirs(memorization_dir, exist_ok=True)
        command_to_run = f"\n ibrun python -m torch.distributed.run --nnodes=$SLURM_JOB_NUM_NODES --rdzv_id=$RANDOM --rdzv_backend=c10d --rdzv_endpoint $head_node_ip:29500 --nproc_per_node=1 find_dataset_neighbors.py --input_dir={outdir} --output_dir={memorization_dir} --features_path=/scratch/07362/gdaras/{dataset}_features.npy --data={dataset_path} 2>&1"
        cmd += f"\n echo '{command_to_run}'"
        # add memorization to the mix
        cmd += f"\n mem_output=$({command_to_run})"
        cmd += f"\n echo 'Memorization done. Parsing results...'"
        cmd += """\n MEAN=$(echo "$mem_output" | grep "Mean of max_products:" | awk '{print $4}')"""
        cmd += """\n MEDIAN=$(echo "$mem_output" | grep "Median of max_products:" | awk '{print $4}')"""
        cmd += """\n PERCENT_99=$(echo "$mem_output" | grep "# > 0.99:" | awk '{print $4}')"""
        cmd += """\n PERCENT_95=$(echo "$mem_output" | grep "# > 0.95:" | awk '{print $4}')"""
        cmd += """\n PERCENT_90=$(echo "$mem_output" | grep "# > 0.9:" | awk '{print $4}')"""
        cmd += f"\n echo 'Computed: MEAN=$MEAN, MEDIAN=$MEDIAN, PERCENT_99=$PERCENT_99, PERCENT_95=$PERCENT_95, PERCENT_90=$PERCENT_90'"
    else:
        cmd += "\n echo 'Skipping memorization'"
        cmd += "\n MEAN=-1.0"
        cmd += "\n MEDIAN=-1.0"
        cmd += "\n PERCENT_99=-1.0"
        cmd += "\n PERCENT_95=-1.0"
        cmd += "\n PERCENT_90=-1.0"

    cmd += "\n echo 'Running FID computation...' \n"
    if "afhq" in dataset_path:
        command_to_run = f'\n python -m torch.distributed.run --standalone eval_fid.py --gen_path={outdir} --ref_path=/scratch/07362/gdaras/datasets/afhqv2-64x64/ 2>&1'

        # evaluate dogness
        # command_to_run = f'\n python -m torch.distributed.run --standalone eval_fid.py --gen_path={outdir} --ref_path=/scratch/07362/gdaras/datasets/afhqv2-64x64-partitioned/1 2>&1'
        # evaluate catness
        # command_to_run = f'\n python -m torch.distributed.run --standalone eval_fid.py --gen_path={outdir} --ref_path=/scratch/07362/gdaras/datasets/afhqv2-64x64-partitioned/0 2>&1'
        # evaluate wildlife
        # command_to_run = f'\n python -m torch.distributed.run --standalone eval_fid.py --gen_path={outdir} --ref_path=/scratch/07362/gdaras/datasets/afhqv2-64x64-part2 2>&1'
    else:
        command_to_run = f'\n python -m torch.distributed.run --standalone eval_fid.py --gen_path={outdir} --ref_stats=${dataset.upper()}_STATS 2>&1'
    cmd += f"\n echo '{command_to_run}'"
    cmd += f"\n output=$({command_to_run})"
    cmd += "\n echo $output "
    cmd += "\n echo 'FID computation done. Parsing results...' \n"
    cmd += """\n FID=$(echo "$output" | grep "FID score:" | awk '{print $3}')"""
    cmd += """\n INCEPTION=$(echo "$output" | grep "Inception score:" | awk '{print $3}')"""
    cmd += """\n echo "Computed: FID:$FID, INCEPTION:$INCEPTION" """

    cmd += f"\n echo 'Writing results to {output_file}'"
    checkpoint_index = checkpoint.split("/")[-1].split("-")[2].split(".")[0]
    cmd += f"""\n echo "Dataset_path={dataset_path}, Checkpoint={checkpoint}, DP={dp}, Checkpoint_index={checkpoint_index}, SEED={seed_id}, FID=$FID, INCEPTION=$INCEPTION, MEAN=$MEAN, MEDIAN=$MEDIAN, PERCENT_99=$PERCENT_99, PERCENT_95=$PERCENT_95, PERCENT_90=$PERCENT_90" >> {output_file}""" 

    # print the whole command
    cmd += f"\n echo '{cmd}'"
    return cmd


def run():
    # parse params from env
    dataset = os.environ.get("DATASET", "cifar")
    dataset_path = os.environ.get("DATASET_PATH", None)
    seed_id = float(os.environ.get("SEED_ID", 0))
    dp = float(os.environ.get("DP", 1.0))
    run_memorization = os.environ.get("RUN_MEMORIZATION", "False") == "True"
    num_generate = int(os.environ.get("NUM_GENERATE", 50000))


    # checkpoints related
    checkpoint_index = os.environ.get("CHECKPOINT_INDEX")
    checkpoint_index = int(checkpoint_index) if checkpoint_index is not None else None
    # fix a pre-trained checkpoint if needed
    pretrained_ckpt = os.environ.get("PRETRAINED_CKPT", None)

    # where to save the eval results
    output_file = os.environ.get("OUTPUT_FILE", f"{dataset.upper()}_EVALS.txt")
    # whether to rerun if generations already exist
    rerun = os.environ.get("RERUN", "False") == "True"

    # optimization params
    weight_decay = float(os.environ.get("WEIGHT_DECAY", 0.0))

    if pretrained_ckpt is None:
        # find training folder
        cmd = 'echo "Looking for training folder"'

        folders = find_training_folders_based_on_params(
            noise_config=None, 
            corruption_probability=0.0, 
            dataset=dataset, 
            dataset_path=dataset_path,
            dataset_keep_percentage=dp,
            weight_decay=weight_decay,
        )

        if folders:
            cmd += f'\n echo "Training folders found: {folders}"'
        else:
            cmd += '\n echo "abort: Training folder not found."'
            print(cmd)
            return

        if checkpoint_index is None:
            # use latest
            checkpoint = find_latest_checkpoint(folders, pkl=True)
        else:
            # find nearest
            checkpoint = find_nearest_checkpoint(folders, checkpoint_index, pkl=True)
    else:
        cmd = f'\n echo "Using pretrained checkpoint: {pretrained_ckpt}"'
        checkpoint = pretrained_ckpt

    if checkpoint is not None:
        cmd += f'\n echo "Checkpoint found: {checkpoint}"'
        cmd += "\n " + get_eval_cmd(seed_id, checkpoint, dataset, dataset_path, dp, output_file, rerun=rerun, run_memorization=run_memorization, num_generate=num_generate)
    else:
        cmd += (
            '\n echo "abort: Checkpoint not found"'
        )

    print(cmd)


if __name__ == "__main__":
    run()
