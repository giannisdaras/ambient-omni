# Frequency based annotation
import os
from slurm_jobs.utils import setup

def get_train_cmd(inference_noise_config, corruption_probability, dataset, dataset_path, checkpoint_paths, flip_probs, annotated_dataset_path, save_only_clean):
    extra_str = "" if not save_only_clean else "-only-clean"
    if annotated_dataset_path is None:
        annotated_dataset_path = os.path.join("/scratch/07362/gdaras/datasets/", 
                                            f"{dataset.lower()}-crops-{inference_noise_config}-{corruption_probability}{extra_str}")
    flip_probs_str = "--flip_probs" if flip_probs else ""
    save_only_clean_str = "--save_only_clean" if save_only_clean else ""
    cmd = f"""ibrun python -m torch.distributed.run \
            --nnodes=$SLURM_JOB_NUM_NODES --rdzv_id=$RANDOM --rdzv_backend=c10d --rdzv_endpoint $head_node_ip:29500 --nproc_per_node=1 annotate_crops.py \
            --annotated_dataset_path={annotated_dataset_path} \
            --inference_noise_config={inference_noise_config} \
            --corruption_probability={corruption_probability} \
            --data={dataset_path} \
            --checkpoint_paths={checkpoint_paths} \
            {save_only_clean_str} \
            {flip_probs_str}"""
    
    return cmd


def run():
    inference_noise_config = os.environ["INFERENCE_NOISE_CONFIG"] if "INFERENCE_NOISE_CONFIG" in os.environ else "identity"
    corruption_probability = float(os.environ["CORRUPTION_PROBABILITY"])
    dataset = os.environ["DATASET"]    
    dataset_path = os.environ["DATASET_PATH"]
    checkpoint_paths = os.environ["CHECKPOINT_PATHS"] 
    flip_probs = os.environ["FLIP_PROBS"] == "True" if "FLIP_PROBS" in os.environ else False
    annotated_dataset_path = os.environ["ANNOTATED_DATASET_PATH"] if "ANNOTATED_DATASET_PATH" in os.environ else None
    save_only_clean = os.environ["SAVE_ONLY_CLEAN"] == "True"
    cmd = "echo Running \n"        
    cmd += get_train_cmd(inference_noise_config, 
        corruption_probability, dataset, dataset_path, 
        checkpoint_paths, flip_probs, annotated_dataset_path, 
        save_only_clean)
    print(cmd)

if __name__ == "__main__":
    run()

