#!/bin/bash
PYTHONPATH=.
dogs_dataset_path=./data/afhqv2-64x64-partitioned/1
ckpt_dir=cats-100_and_dogs-100/network-snapshot-020070/s-max-4/00009-network-snapshot-020070-uncond-ddpmpp-edm-gpus8-batch512-fp32-FtdC1
dp=1.0
weight_decay=0.0

iters_list=("005018" "010035" "015053" "020070" "025088" "030106" "035124" "040141") # "045159") #() # "006021" "007025" "008028" "009032" #("011039" "012042" "013046" "014049"  "016056" "017060" "018063" "019067" ) # "025088"  "050176" "075264" "100352""125440" "150528" "175616" "200000")

# Double for loop to iterate over both lists
for iter_num in "${iters_list[@]}"; do
  ckpt_name=${ckpt_dir}/network-snapshot-$iter_num
  ckpt_path=./outputs/ambient-syn-runs/afhq-64x64/ood_diffusion/cats_help_dogs/${ckpt_name}.pkl
  eval_path=./outputs/ambient-syn-evals/afhq-64x64/ood_diffusion/cats_help_dogs/${ckpt_name}

  mkdir -p $eval_path

  # Randomize torchrun master_port
  MASTER_PORT=$(( ( RANDOM % 1000 )  + 10000 ))

  echo "Using pretrained checkpoint: $ckpt_path"
  echo "Checkpoint found: $ckpt_path"

  # Generate
  torchrun --master_port $MASTER_PORT --nproc_per_node=8 generate.py --seeds=0-5238 --network=$ckpt_path \
  --outdir=$eval_path  --steps=18

  # FID
  output=$(torchrun --standalone eval_fid.py --gen_path=$eval_path --ref_path=$dogs_dataset_path)
  FID=$(echo "$output" | grep "FID score:" | awk '{print $3}')
  INCEPTION=$(echo "$output" | grep "Inception score:" | awk '{print $3}')
  echo "Dataset=cifar, Checkpoint=$ckpt_path, DP=1.0, Checkpoint_index=070246, SEED=0.0, FID=$FID, INCEPTION=$INCEPTION"
  echo "Dataset=cifar, Checkpoint=$ckpt_path, DP=1.0, Checkpoint_index=070246, SEED=0.0, FID=$FID, INCEPTION=$INCEPTION" >> $eval_path/eval.txt

  # # FD DINOv2
  # cd ../edm2
  # output=$(python calculate_metrics.py calc --images=$eval_path --ref=./data/annotated_cifar10_uncorrupted/refs.pkl)
  # echo "Dataset=cifar, Checkpoint=$ckpt_path, Calc=$output"
  # echo "Dataset=cifar, Checkpoint=$ckpt_path, Calc=$output" >> $eval_path/eval_fd_dino.txt
  # cd ../pixel-diffusion
done