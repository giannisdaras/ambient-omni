# Frequency based annotation
import os
from slurm_jobs.utils import setup

def get_train_cmd(inference_noise_config, corruption_probability, dataset, dataset_path, min_fixed_sigma, max_fixed_sigma, save_only_clean):
    extra_str = "" if not save_only_clean else "-only-clean"
    annotated_datasets_path = os.path.join("/scratch/07362/gdaras/datasets/", 
                                           f"{dataset.lower()}-fixed-sigma-{min_fixed_sigma}-{max_fixed_sigma}-{inference_noise_config}-{corruption_probability}{extra_str}")
    save_only_clean_str = "--save_only_clean" if save_only_clean else ""
    cmd = f"""ibrun python -m torch.distributed.run \
            --nnodes=$SLURM_JOB_NUM_NODES --rdzv_id=$RANDOM --rdzv_backend=c10d --rdzv_endpoint $head_node_ip:29500 --nproc_per_node=1 annotate_fixed_sigma.py \
            --annotated_dataset_path={annotated_datasets_path} \
            --inference_noise_config={inference_noise_config} \
            --corruption_probability={corruption_probability} \
            --min_fixed_sigma={min_fixed_sigma} \
            --max_fixed_sigma={max_fixed_sigma} \
            --data={dataset_path} \
            {save_only_clean_str}"""
    
    return cmd


def run():
    inference_noise_config = os.environ["INFERENCE_NOISE_CONFIG"] if "INFERENCE_NOISE_CONFIG" in os.environ else "blurs3"
    corruption_probability = float(os.environ["CORRUPTION_PROBABILITY"])
    dataset = os.environ["DATASET"]    
    dataset_path = os.environ["DATASET_PATH"]
    min_fixed_sigma = float(os.environ["MIN_FIXED_SIGMA"])
    max_fixed_sigma = float(os.environ["MAX_FIXED_SIGMA"])
    save_only_clean = os.environ["SAVE_ONLY_CLEAN"] == "True"
    cmd = "echo Running \n"        
    cmd += get_train_cmd(inference_noise_config, corruption_probability, dataset, dataset_path, min_fixed_sigma, max_fixed_sigma, save_only_clean)
    print(cmd)

if __name__ == "__main__":
    run()

