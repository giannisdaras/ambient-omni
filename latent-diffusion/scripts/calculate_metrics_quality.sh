data=./scratch_release/datasets/img512.zip
dest=./scratch_release/annotations/clip_iqa_patch_average

# Calculate quality
mkdir -p "$(dirname "$dest")"
torchrun --nproc_per_node=8 calculate_metrics_quality.py ref --data=$data --dest=$dest --batch=128

# Delete per-node files
# rm -f $dest_{0..7}.pkl