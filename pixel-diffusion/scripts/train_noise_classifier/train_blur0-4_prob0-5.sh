
#!/bin/bash
PYTHONPATH=.
outdir=./outputs # Replace with your own
dataset_path=/data/vision/torralba/selfmanaged/torralba/projects/adrianr/ambient/data_softlink/cifar10/train # Replace with your own
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