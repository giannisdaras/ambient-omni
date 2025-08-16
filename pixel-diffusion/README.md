
# Ambient Pixel Diffusion

This part of the repo focuses on small-scale experiments pixel diffusion with [EDM](https://github.com/NVlabs/edm). There are two things you can do here:
1. Train a generative model that leverages low-quality images (e.g. blurry images, JPEG compressed, etc)
2. Train a generative model using out-of-distribution samples (e.g. a generative model for dogs that also uses images of cats)


## 1. Train a high-quality generative model for cifar-10 using mostly low-quality corrupted data

To do this, there are four steps you must follow:

<ol type="a">
<li><a href="#1a-train-a-noise-classifier-to-distinguish-noisy-clean-images-from-noisy-corrupted-images">Train a noise classifier to distinguish noisy clean images from noisy corrupted images</a></li>
<li><a href="#1b-annotate-a-mostly-corrupted-dataset-with-minimum-noise-levels-sigma_tn">Annotate a (mostly) corrupted dataset with minimum noise levels $\sigma_{tn}$</a></li>
<li><a href="#1c-train-a-generative-model-with-the-annotated-corrupted-dataset-using-ambient-diffusion">Train a generative model with the annotated corrupted dataset using ambient diffusion</a></li>
<li><a href="#1d-evaluate-the-generative-model">Evaluate the generative model</a></li>
</ol>

### 1.a [Optional] Train a noise classifier to distinguish noisy clean images from noisy corrupted images

Training a classifier is the most principled way of using our method, and leads to the best results. However, this is optional as using a fixed hyper-parameter annotation will also yield good results. If you want to skip this step, go ahead to [1.b](https://github.com/giannisdaras/ambient-omni/blob/main/pixel-diffusion/README.md#1b-annotate-a-mostly-corrupted-dataset-with-minimum-noise-levels-sigma_tn), otherwise keep reading. Training a classifier can be done using the scripts in `scripts/train_noise_classifier`, with an example shown below for blurring corruptions with $\sigma_B=0.4$. Note that you will have to replace the `dataset_path` to cifar10 with your own
```
# train_blur0-4_prob0-5.sh
#!/bin/bash
PYTHONPATH=.
outdir=./outputs
dataset_path=./data/cifar10/train # Replace with your own
noise_config=blur0-4
corruption_probability=0.5
precond=edmcls
dp=1.0
weight_decay=0.0

# Set experiment dir
outdir=${outdir}/ambient-syn-runs/cifar/noise_classifier/${noise_config}_prob${corruption_probability//./-}

# Randomize torchrun master_port
MASTER_PORT=$(( ( RANDOM % 1000 )  + 10000 ))

mkdir -p $outdir
torchrun --master_port $MASTER_PORT --nproc_per_node=8 train.py \
            --outdir=${outdir} \
            --data=${dataset_path} \
            --cond=0 --arch=ddpmpp --dump=40 --duration=200 --lr=1e-4 \
            --precond=${precond} \
            --noise_config=${noise_config} \
            --corruption_probability=${corruption_probability} \
            --dataset_keep_percentage=${dp} \
            --weight_decay=${weight_decay} \
            --workers=2 \
            --expr_id=train_noise_classifier_${noise_config}_prob${corruption_probability//./-}_cifar10_dp${dp}_wd${weight_decay}
```

### 1.b Annotate a (mostly) corrupted dataset with minimum noise levels $\sigma_{tn}$

This can be done using the scripts in `scripts/annotate_noise_classifier` (if using a classifier) and `scripts/annotate_fixed_sigma` (if using a fixed annotation), with examples shown below for blurring corruptions with $\sigma_B=0.8$ and $\sigma_B=0.4$. Note that you will have to replace the `ckpt_name`, `annotated_datasets_path`, and `checkpoint_path` with your own. We have also uploaded a classifier checkpoint to [huggingface](https://huggingface.co/adrianrm/ambient-o-noise-classifier-blur06-prob05-iter15k) for blurring corruptions with $\sigma_B=0.6$, to help sanity check any experiments (our checkpoint requires the cifar10 data to be in `./data/cifar10/train`).

With fixed sigma
```
# blur0-8_fixed1-89.sh
#!/bin/bash
PYTHONPATH=.
annotated_datasets_path=./outputs/annotated_cifar10/annotated_blur0-8_fixed_1-89/
training_noise_config=blur0-8
inference_noise_config=blur0-8
corruption_probability=0.9
min_sigma=1.89

# Randomize torchrun master_port
MASTER_PORT=$(( ( RANDOM % 1000 )  + 10000 ))

export TORCH_NCCL_ENABLE_MONITORING=0
# export NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=72000

torchrun --nproc_per_node=1 annotate_fixed_sigma.py \
    --annotated_dataset_path=${annotated_datasets_path} \
    --dataset_path=./data/cifar10/train \
    --inference_noise_config=${inference_noise_config} \
    --corruption_probability=${corruption_probability} \
    --min_fixed_sigma=$min_sigma \
    --max_fixed_sigma=0
```

With your own classifier
```
# blur0-4_prob0-5_15k.sh
#!/bin/bash
PYTHONPATH=.
ckpt_name=noise_classifier/blur0-4_prob0-5/00004-train-uncond-ddpmpp-edmcls-gpus8-batch512-fp32-V5gDx/network-snapshot-015053 # Replace with your own
annotated_datasets_path=./outputs/annotated_cifar10/$ckpt_name # Replace with your own
checkpoint_path=./outputs/ambient-syn-runs/cifar/$ckpt_name.pkl # Replace with your own
training_noise_config=blur0-4
inference_noise_config=blur0-4
corruption_probability=0.9

# Randomize torchrun master_port
MASTER_PORT=$(( ( RANDOM % 1000 )  + 10000 ))

export TORCH_NCCL_ENABLE_MONITORING=0
# export NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=72000

mkdir -p $annotated_datasets_path
torchrun --nproc_per_node=8 annotate.py \
    --annotated_dataset_path=${annotated_datasets_path} \
    --training_noise_config=${training_noise_config} \
    --inference_noise_config=${inference_noise_config} \
    --corruption_probability=${corruption_probability} \
    --checkpoint_path=${checkpoint_path}
```

With our huggingface checkpoint. Using our checkpoint requires the cifar10 data to be in `./data/cifar10/train`, or that you add a line changing the `dataset_kwargs` in line 51 of `annotate.py`. 
```
#!/bin/bash
PYTHONPATH=.
ckpt_name=noise_classifier/blur0-6_prob0-5/00000-train-uncond-ddpmpp-edmcls-gpus8-batch512-fp32-TWeeb/network-snapshot-015053 # Replace with your own
annotated_datasets_path=./outputs/annotated_cifar10/$ckpt_name # Replace with your own
checkpoint_path=adrianrm/ambient-o-noise-classifier-blur06-prob05-iter15k # Using huggingface checkpoint
training_noise_config=blur0-6
inference_noise_config=blur0-6
corruption_probability=0.9

# Randomize torchrun master_port
MASTER_PORT=$(( ( RANDOM % 1000 )  + 10000 ))

export TORCH_NCCL_ENABLE_MONITORING=0
# export NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=72000

mkdir -p $annotated_datasets_path
torchrun --nproc_per_node=7 annotate.py \
    --annotated_dataset_path=${annotated_datasets_path} \
    --training_noise_config=${training_noise_config} \
    --inference_noise_config=${inference_noise_config} \
    --corruption_probability=${corruption_probability} \
    --checkpoint_path=${checkpoint_path}
```


### 1.c Train a generative model with the annotated corrupted dataset using ambient diffusion

This can be done using the scripts in `scripts/train_low_quality_data_diffusion`, with an example shown below for blurring corruptions with $\sigma_B=0.4$. Note that you will have to replace the `dataset_path` to the annotated dataset from 2.b for your own.

```
# train_blur0-4_prob0-5_iter15k.sh
#!/bin/bash
PYTHONPATH=.
outdir=./outputs
# Replace with your own; path to annotated dataset
dataset_path=${outdir}/annotated_cifar10/noise_classifier/blur0-4_prob0-5/00004-train-uncond-ddpmpp-edmcls-gpus8-batch512-fp32-V5gDx/network-snapshot-015053
s_max=4
dp=1.0
weight_decay=0.0
cls_epsilon=0.05

# Set experiment dir
save_path=$(echo "$dataset_path" | sed 's|.*/annotated_cifar10/||')
outdir=${outdir}/ambient-syn-runs/cifar/low_quality_data_diffusion/${save_path}/s-max-${s_max}

# Extract blur and prob parameters
blur_config=$(echo "$dataset_path" | sed -n 's|.*blur\([0-9-]*\)_prob.*|\1|p')
prob_config=$(echo "$dataset_path" | sed -n 's|.*prob\([0-9-]*\).*|\1|p')

# Randomize torchrun master_port
MASTER_PORT=$(( ( RANDOM % 1000 )  + 10000 ))

mkdir -p $outdir
torchrun --master_port $MASTER_PORT --nproc_per_node=8 train.py \
            --outdir=${outdir} \
            --data=${dataset_path} \
            --cond=0 --arch=ddpmpp --dump=50 --duration=200 \
            --precond=edm \
            --corruption_probability=0.0 \
            --dataset_keep_percentage=${dp} \
            --weight_decay=${weight_decay} \
            --workers 2 \
            --cls_epsilon=${cls_epsilon} \
            --expr_id=train_low_quality_data_diffusion_cifar10_blur${blur_config}_prob${prob_config}_s-max-${s_max}_dp${dp}_wd${weight_decay} \
            --s_max=${s_max}
```

### 1.d Evaluate the generative model

This can be done using the scripts in `scripts/eval_low_quality_data_diffusion`, with an example shown below for blurring corruptions with $\sigma_B=0.4$. Note that you will have to replace the `dataset_path` to the clean cifar10 and the `ckpt_dir` of your trained generative model from 1.c.

```
# eval_blur0-4_prob0-5_iter15k.sh
#!/bin/bash
PYTHONPATH=.
dataset_path=./data/train # Replace with your own; path to clean cifar10
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
```
## 2. Train a generative model for dogs utilising images of cats

To do this, there are five steps you must follow:
<ol type="a">
<li><a href="#2a-merge-the-datasets-of-cats-and-dogs-into-one-folder">Merge the datasets of cats and dogs into one folder</a></li>
<li><a href="#2b-train-a-crops-classifier-to-distinguish-crops-from-dog-and-cat-images">Train a crops classifier to distinguish crops from dog and cat images</a></li>
<li><a href="#2c-annotate-a-mostly-out-of-distribution-dataset-with-maximum-crop-size">Annotate a (mostly) out-of-distribution dataset with maximum crop size</a></li>
<li><a href="#2d-train-a-generative-model-for-dogs-utilising-images-of-cats">Train a generative model for dogs utilising images of cats</a></li>
<li><a href="#2e-evaluate-the-generative-model">Evaluate the generative model</a></li>
</ol>

### 2.a Merge the datasets of cats and dogs into one folder
This can be done using the script `scripts/ood_dataset_utils/merge_cats_and_dogs_for_crops_classifier.sh`. Note that you will have to replace both `cats_dataset_path` and the `dogs_dataset_path` with your own
```
# merge_cats_and_dogs_for_crops_classifier.sh
cats_dataset_path=./data/afhqv2-64x64-partitioned/0 # cats, replace with your own
dogs_dataset_path=./data/afhqv2-64x64-partitioned/1 # dogs, replace with your own
./scripts/ood_dataset_utils/merge_datasets_for_train_crops_classifier.sh ./data/cats-100_and_dogs-100 $cats_dataset_path $dogs_dataset_path
```

### 2.b Train a crops classifier to distinguish crops from dog and cat images

This can be done using the scripts in `scripts/train_crops_classifier` with an example shown below for cats and dogs. Note that you will have to run all crop size (4, 8, 16, 24), not just one of them.
```
# dogs_and_cats_4.sh
#!/bin/bash
PYTHONPATH=.
outdir=./outputs
dataset_path=./data/cats-100_and_dogs-100
crop_size=4
dp=1.0
weight_decay=0.0

# Add args to outdir
dataset_name=$(basename "$dataset_path")
outdir=${outdir}/ambient-syn-runs/afhq-64x64/crops_classifier/${dataset_name}/crop-${crop_size}

# Randomize torchrun master_port
MASTER_PORT=$(( ( RANDOM % 1000 )  + 10000 ))

mkdir -p $outdir
torchrun --master_port $MASTER_PORT --nproc_per_node=8 train.py \
            --outdir=${outdir} \
            --data=${dataset_path} \
            --cond=0 --arch=ddpmpp --dump=50 --duration=200 \
            --precond=edmcls \
            --noise_config=identity \
            --corruption_probability=0 \
            --dataset_keep_percentage=${dp} \
            --weight_decay=${weight_decay} \
            --overwrite_cls_labels_path=${dataset_path}/labels.jsonl \
            --crop_size=${crop_size} \
            --expr_id=train_crops_classifier_dogs_and_cats_dp${dp}_wd${weight_decay} \
            --workers=2
```

### 2.c Annotate a (mostly) out-of-distribution dataset with maximum crop size

This can be done using the scripts in `scripts/annotate_crops_classifier` with an example shown below. Note that you will have to replace `cats_dataset_path`, `dogs_dataset_path`, `checkpoint_path_4`, `checkpoint_path_8`, `checkpoint_path_16`, and `checkpoint_path_24` with your own.
```
# annotate_crops_cat_is_dog_20k.sh
#!/bin/bash
PYTHONPATH=.

cats_dataset_path=./data/afhqv2-64x64-partitioned/0 # cats, replace with your own
dogs_dataset_path=./data/afhqv2-64x64-partitioned/1 # dogs, replace with your own
checkpoint_path=./outputs/ambient-syn-runs/afhq-64x64/crops_classifier/cats-100_and_dogs-100
# Replace with your own
checkpoint_path_4=${checkpoint_path}/crop-4/00000-cats-100_and_dogs-100-uncond-ddpmpp-edmcls-gpus8-batch512-fp32-56WOn/network-snapshot-020070.pkl
checkpoint_path_8=${checkpoint_path}/crop-8/00000-cats-100_and_dogs-100-uncond-ddpmpp-edmcls-gpus8-batch512-fp32-0A6eU/network-snapshot-020070.pkl
checkpoint_path_16=${checkpoint_path}/crop-16/00000-cats-100_and_dogs-100-uncond-ddpmpp-edmcls-gpus8-batch512-fp32-lkjKu/network-snapshot-020070.pkl
checkpoint_path_24=${checkpoint_path}/crop-24/00000-cats-100_and_dogs-100-uncond-ddpmpp-edmcls-gpus8-batch512-fp32-g6Cgu/network-snapshot-020070.pkl
checkpoint_paths=${checkpoint_path_4},${checkpoint_path_8},${checkpoint_path_16},${checkpoint_path_24}

# Wrangle paths auto
checkpoint_basename=$(basename "$checkpoint_path")
checkpoint_path_network_snapshot=$(basename "$checkpoint_path_4" .pkl)
annotated_dataset_path=./outputs/annotated_afhq/cat_is_dog/${checkpoint_basename}/${checkpoint_path_network_snapshot}

# Annotate cat crops
mkdir -p $annotated_dataset_path
torchrun --nproc_per_node=8 annotate_crops.py \
          --annotated_dataset_path=${annotated_dataset_path} \
          --inference_noise_config=identity \
          --corruption_probability=0 \
          --data=${cats_dataset_path} \
          --checkpoint_paths=${checkpoint_paths}

# Merge cat crops dataset with 10% dogs data
merged_dataset_path=./outputs/annotated_afhq/cats_help_dogs/${checkpoint_basename}/${checkpoint_path_network_snapshot}
./scripts/ood_dataset_utils/merge_datasets_for_train_ood_diffusion.sh ${merged_dataset_path} ${dogs_dataset_path}:10 ${annotated_dataset_path}:100
```

### 2.d Train a generative model for dogs utilising images of cats

This can be done using the scripts in `scripts/train_ood_diffusion` with an example shown below. Note that you will have to replace the `dataset_path` to the annotated dataset with your own.
```
# train_cats_help_dogs_iter20k.sh
#!/bin/bash
PYTHONPATH=.
outdir=./outputs
dataset_path=./outputs/annotated_afhq/cats_help_dogs/cats-100_and_dogs-100/network-snapshot-020070 # Replace with your own; path to annotated dataset
s_max=4
dp=1.0
weight_decay=0.0
cls_epsilon=0.05

extracted_path=$(echo "$dataset_path" | sed 's/.*annotated_afhq\///')
outdir=${outdir}/ambient-syn-runs/afhq-64x64/ood_diffusion/${extracted_path}/s-max-${s_max}
config_part=$(basename "$(dirname "$dataset_path")")

# Randomize torchrun master_port
MASTER_PORT=$(( ( RANDOM % 1000 )  + 10000 ))

torchrun --master_port $MASTER_PORT --nproc_per_node=8 train.py \
            --outdir=${outdir} \
            --data=${dataset_path} \
            --cond=0 --arch=ddpmpp --dump=40 --duration=200 \
            --precond=edm \
            --corruption_probability=0.0 \
            --dataset_keep_percentage=${dp} \
            --weight_decay=${weight_decay} \
            --workers 2 \
            --cls_epsilon=${cls_epsilon} \
            --expr_id=train_ood_diffusion_afhq_cats_help_dogs_${config_part}_s-max-${s_max}_dp${dp}_wd${weight_decay} \
            --s_max=${s_max}
```

### 2.e Evaluate the generative model

This can be done using the scripts in `scripts/eval_ood_diffusion` with an example shown below. Note that you will have to replace the `ckpt_dir` to the annotated dataset with your own.
```
# eval_half_cats_help_dogs_iter15k.sh
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
```

# 🔗 Related Codebases

* [EDM](https://github.com/NVlabs/edm): starting point for this repository.
* [Ambient utils](https://github.com/giannisdaras/ambient-utils): helper functions for training diffusion models (or flow matching models) in settings with limited access to high-quality data.
* [Ambient Laws](https://github.com/giannisdaras/ambient-laws): trains models with a mix of clean and noisy data.
* [Ambient Diffusion](https://github.com/giannisdaras/ambient-diffusion): trains models for linear corruptions.
* [Consistent Diffusion Meets Tweedie](https://github.com/giannisdaras/ambient-tweedie): trains models with only noisy data, with support for Stable Diffusion finetuning.
* [Consistent Diffusion Models](https://github.com/giannisdaras/cdm): original implementation of the consistency loss.


# 📧 Contact

If you are interested in colaborating, please reach out to gdaras[at]mit[dot]edu.


