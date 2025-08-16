# Paths and parameters (customize these)
output_path="./outputs/generations/diffdb-vendi/ambient-o"
model_path=giannisdaras/ambient-o
guidance_scale=2.00
num_inference_steps=30
dataset_path="./datadir/diffdb/mds/"

output_path=${output_path}/guidance-scale-${guidance_scale}_num-steps-${num_inference_steps}

# Launch
mkdir -p $output_path
torchrun --nproc_per_node=8 generate_many_from_dataset_captions_and_measure_vendi.py \
    --output_path=${output_path} \
    --model_path=${model_path} \
    --guidance_scale=${guidance_scale} \
    --num_inference_steps=${num_inference_steps} \
    --dataset_path=${dataset_path}
