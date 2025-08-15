
#!/bin/bash
PYTHONPATH=.
outdir=/data/vision/torralba/selfmanaged/torralba/scratch/adrianr/ambient/ambient-syn-runs/cifar/aaa-test_classifier_half_blur06_prob05
dataset_path=/data/vision/torralba/selfmanaged/torralba/projects/adrianr/ambient/data_softlink/cifar10/train
precond=edmcls-half
noise_config=blurs_06
corruption_probability=0.5
dp=1.0
weight_decay=0.0
all_pairs=False

# Randomize torchrun master_port
MASTER_PORT=$(( ( RANDOM % 1000 )  + 10000 ))

torchrun --master_port $MASTER_PORT --nproc_per_node=8 train.py \
            --outdir=${outdir} \
            --data=${dataset_path} \
            --cond=0 --arch=ddpmpp --dump=50 --duration=200 --lr=1e-4 \
            --precond=${precond} \
            --noise_config=${noise_config} \
            --corruption_probability=${corruption_probability} \
            --dataset_keep_percentage=${dp} \
            --weight_decay=${weight_decay} \
            --all_pairs=${all_pairs} \
            --workers=2 \
            --keep_schedule=False \
            --expr_id=train_cls_${noise_config}_cifar10_dp${dp}_wd${weight_decay}_all_pairs${all_pairs}