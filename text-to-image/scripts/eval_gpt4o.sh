#!/usr/bin/env bash

folderA= # generated images of model A
folderB= # generated images of model B
path_to_prompts='./datadir/drawbench_prompts.csv'

results_dir='./outputs/evaluations/gpt4o/drawbench'
results_name= # name of experiment

export OPENAI_API_KEY=

mkdir -p $results_dir
torchrun --nproc_per_node=1 gpt4o_eval_compare.py \
  ${folderA} \
  ${folderB} \
  --prompts_file ${path_to_prompts}  \
  --output ${results_dir}/${results_name}.csv