# Args
experiment_dir=outputs/ambient-syn-edm2-runs/edm2-img512-xxl-ambient-o/bad_data_percentage-0.8-bad-data-sigma-min-0.2-use-ambient-crops-True

## Ema
ema_std=0.015
kimg=939524

## Guidance
gnet=https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-xs-uncond-2147483-0.015.pkl
guidance=1.20

## Seeds
seed_min=50000
seed_max=99999

# Evaluation dir
experiment_name=${experiment_dir##*/ambient-syn-edm2-runs/}
outdir=outputs/ambient-syn-edm2-evals/$experiment_name

# # Compute phema
# ema_dir=${experiment_dir}-ema-${ema_std}
# python reconstruct_phema.py --indir=$experiment_dir --outdir={ema_dir} --outstd={ema_std} --outkimg={kimg}

# Generate images
net=giannisdaras/ambient-o-imagenet512-xxl-with-crops #$ema_dir/phema-${kimg}-${ema_std}.pkl
images_dir=$outdir/ema-${ema_std}-guidance-${guidance}/seeds-${seed_min}-${seed_max}

mkdir -p $images_dir
torchrun --nproc_per_node=8 generate_images.py --net=${net} --outdir=${images_dir} --gnet=${gnet} --guidance=${guidance} --seeds=${seed_min}-${seed_max} --batch 32

# Calculate FID, DINO_FD
python calculate_metrics.py calc --images=${images_dir} --ref=https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl

