#!/bin/bash
# 8-GPU torchrun script for EDM2 training
# Usage: ./run_edm2_8gpu.sh
set -e # Exit on any error

# EDM2 config
model_name="edm2-img512-xs"
dataset_path="datasets/img512-sd.zip"

# Ambient config
annotations_qualities_path="adrianrm/ambient-o-clip-iqa-patches-imagenet" #"annotations/clip_iqa_patch_average.pkl"
bad_data_percentage=0.9
bad_data_sigma_min=0.2
use_ambient_crops=False

# Outdir / id
outdir=outputs/ambient-syn-edm2-runs/${model_name}-ambient-o/bad_data_percentage-${bad_data_percentage}-bad-data-sigma-min-${bad_data_sigma_min}-use-ambient-crops-${use_ambient_crops}
expr_id=${model_name}-ambient-o-bad_data_percentage-${bad_data_percentage}-bad-data-sigma-min-${bad_data_sigma_min}-use-ambient-crops-${use_ambient_crops}

# Create output directory if it doesn't exist
mkdir -p "$outdir"

# Log configuration
echo "Starting EDM2 training with the following configuration:"
echo " EDM Config:"
echo " Model: $model_name"
echo " Dataset: $dataset_path"
echo ""

echo " Ambient config:"
echo " Annotations: $annotations_qualities_path"
echo " Bad data percentage: $bad_data_percentage"
echo " Bad data sigma min: $bad_data_sigma_min"
echo " Use ambient crops: $use_ambient_crops"
echo ""

echo " Saving to:"
echo " Output: $outdir"
echo " Experiment ID: $expr_id"
echo ""

# Run distributed training with torchrun
torchrun \
    --nproc_per_node=8 \
    train_edm2.py \
    --preset="$model_name" \
    --data="$dataset_path" \
    --annotations_qualities_path="$annotations_qualities_path" \
    --bad_data_percentage=$bad_data_percentage \
    --bad_data_sigma_min=$bad_data_sigma_min \
    --use_ambient_crops=$use_ambient_crops \
    --outdir="$outdir" \
    --expr_id="$expr_id"

echo ""
echo "Training completed. Check output directory: $outdir"