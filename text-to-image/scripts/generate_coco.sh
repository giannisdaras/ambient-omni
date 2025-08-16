# Paths and parameters (customize these)
output_path="./outputs/generations/coco/ambient-o"
model_path=giannisdaras/ambient-o
guidance_scale=1.50
num_inference_steps=30
dataset_captions_path="./datadir/MS-COCO_val2014_30k_captions.csv" # https://github.com/boomb0om/text2image-benchmark/tree/main?tab=readme-ov-file
batch_size=16

output_path=${output_path}/guidance-scale-${guidance_scale}_num-steps-${num_inference_steps}

# Launch
mkdir -p $output_path
torchrun --nproc_per_node=8 generate_from_caption_file.py \
    --output_path=${output_path} \
    --model_path=${model_path} \
    --guidance_scale=${guidance_scale} \
    --num_inference_steps=${num_inference_steps} \
    --dataset_captions_path=${dataset_captions_path} \
    --batch_size=${batch_size}
