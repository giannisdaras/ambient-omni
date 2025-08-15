import os
from slurm_jobs.utils import setup, find_training_folders_based_on_params, find_latest_checkpoint, find_nearest_checkpoint

def get_train_cmd(training_noise_config, inference_noise_config, corruption_probability, dataset, checkpoint_path):
    # get id from the checkpoint path
    checkpoint_id = checkpoint_path.split("/")[-2].split("-")[-1]
    iteration = checkpoint_path.split("/")[-1].split("-")[-1].split(".")[0]
    annotated_datasets_path = os.path.join("/scratch/07362/gdaras/datasets/", 
                                           f"{dataset.lower()}-{training_noise_config}-{inference_noise_config}-{corruption_probability}-{checkpoint_id}-{iteration}")
    cmd = f"""ibrun python -m torch.distributed.run \
            --nnodes=$SLURM_JOB_NUM_NODES --rdzv_id=$RANDOM --rdzv_backend=c10d --rdzv_endpoint $head_node_ip:29500 --nproc_per_node=1 annotate.py \
            --annotated_dataset_path={annotated_datasets_path} \
            --training_noise_config={training_noise_config} \
            --inference_noise_config={inference_noise_config} \
            --corruption_probability={corruption_probability} \
            --checkpoint_path={checkpoint_path}"""
    return cmd

def run():
    training_noise_config = os.environ["TRAINING_NOISE_CONFIG"] if "TRAINING_NOISE_CONFIG" in os.environ else "blurs"
    inference_noise_config = os.environ["INFERENCE_NOISE_CONFIG"] if "INFERENCE_NOISE_CONFIG" in os.environ else "blurs3"
    corruption_probability = float(os.environ["CORRUPTION_PROBABILITY"])
    dataset = os.environ["DATASET"]
    dataset_path = os.environ["DATASET_PATH"]

    # automatically find the checkpoint
    checkpoint_index = os.environ.get("CHECKPOINT_INDEX")
    checkpoint_index = int(checkpoint_index) if checkpoint_index is not None else None
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", None)

    if checkpoint_path is None:
        # find training folder
        cmd = 'echo "Looking for training folder" with params: \n'
        cmd += f'noise_config={training_noise_config}, corruption_probability={0.5}, dataset={dataset}, dataset_path={dataset_path}, dataset_keep_percentage={1.0}, weight_decay={0.0} \n'

        folders = find_training_folders_based_on_params(
            noise_config=training_noise_config, 
            corruption_probability=0.5,  # half of the dataset is good and half is bad.
            dataset=dataset, 
            dataset_path=dataset_path,
            dataset_keep_percentage=1.0,
            weight_decay=0.0,
        )

        if folders:
            cmd += f'\n echo "Training folders found: {folders}"'
        else:
            cmd += '\n echo "abort: Training folder not found."'
            print(cmd)
            return

        if checkpoint_index is None:
            # use latest
            checkpoint_path = find_latest_checkpoint(folders, pkl=True)
        else:
            # find nearest
            checkpoint_path = find_nearest_checkpoint(folders, checkpoint_index, pkl=True)

    cmd = f'\n echo "Using pretrained checkpoint: {checkpoint_path}"'
    cmd = "\n echo Running \n"        
    cmd += get_train_cmd(training_noise_config, inference_noise_config, corruption_probability, dataset, checkpoint_path)
    print(cmd)

if __name__ == "__main__":
    run()

