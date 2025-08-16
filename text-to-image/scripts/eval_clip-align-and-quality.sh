#!/usr/bin/env bash

path_to_images= # Path to generated images
path_to_prompts='./datadir/MS-COCO_val2014_30k_captions.csv'
path_to_results='./outputs/evaluations/clip-align-and-quality' # Add model name to path

mkdir -p $path_to_results
torchrun --nproc_per_node=8 clip_eval_openclip.py \
  --images ${path_to_images} \
  --prompts ${path_to_prompts} \
  --output_csv ${path_to_results}/results.csv
