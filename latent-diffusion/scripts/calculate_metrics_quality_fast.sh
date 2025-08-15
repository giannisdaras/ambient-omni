data=datasets/img512.zip
dest=annotations/clip_iqa_patch_average_fast

# Calculate quality
mkdir -p "$(dirname "$dest")"
torchrun --nproc_per_node=8 calculate_metrics_quality.py ref --data=$data --dest=$dest --batch=128 --metrics=CLIP-IQA,CLIP-IQA-512,CLIP-IQA-256

# Delete per-node files
rm -f $dest_{0..7}.pkl