#!/bin/bash
PYTHONPATH=.
dataset_path=./data/cifar10/train # Replace with your own; path to clean cifar10
# Replace with your own
ckpt_dir=noise_classifier/blur0-4_prob0-5/00004-train-uncond-ddpmpp-edmcls-gpus8-batch512-fp32-V5gDx/network-snapshot-015053/s-max-4/00000-network-snapshot-015053-uncond-ddpmpp-edm-gpus8-batch512-fp32-DusSj
dp=1.0
weight_decay=0.0

iters_list=("025088" "050176" "075264" "100352" "125440" "150528" "175616" "200000") #("025088" "050176" "075264" "100352" "125440" 150528" "175616" "200000") #("010035" "011039" "012042" "013046" "014049" "015053" "016056" "017060" "018063" "019067" "020070") #"005018" "006021" "007025" "008028" "009032"  #("005018" "006021" "007025" "008028" "009032" "010035") # 

# Double for loop to iterate over both lists
for iter_num in "${iters_list[@]}"; do
  ckpt_name=${ckpt_dir}/network-snapshot-$iter_num
  ckpt_path=./outputs/ambient-syn-runs/cifar/low_quality_data_diffusion/${ckpt_name}.pkl
  eval_path=./outputs/ambient-syn-evals/cifar/low_quality_data_diffusion/${ckpt_name}

  mkdir -p $eval_path

  # Randomize torchrun master_port
  MASTER_PORT=$(( ( RANDOM % 1000 )  + 10000 ))

  echo "Using pretrained checkpoint: $ckpt_path"
  echo "Checkpoint found: $ckpt_path"

  # Generate
  torchrun --master_port $MASTER_PORT --nproc_per_node=8 generate.py --seeds=0-49999 --network=$ckpt_path \
  --outdir=$eval_path  --steps=18

  # FID
  output=$(torchrun --standalone eval_fid.py --gen_path=$eval_path --ref_path=$dataset_path)
  FID=$(echo "$output" | grep "FID score:" | awk '{print $3}')
  INCEPTION=$(echo "$output" | grep "Inception score:" | awk '{print $3}')
  echo "Dataset=cifar, Checkpoint=$ckpt_path, FID=$FID, INCEPTION=$INCEPTION"
  echo "Dataset=cifar, Checkpoint=$ckpt_path, FID=$FID, INCEPTION=$INCEPTION" >> $eval_path/eval.txt
done