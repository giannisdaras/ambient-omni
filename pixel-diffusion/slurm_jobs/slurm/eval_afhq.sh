slurmify submit-parametric-array \
  --account CGAI24022 \
  --job-name eval_afhq \
  --script-path slurm_jobs/eval_net.py \
  --time-limit 00:15:00 \
  --parameter "dataset:afhq" \
  --parameter "noise_config:identity" \
  --parameter "dataset_path:/scratch/07362/gdaras/datasets/afhq-wildlife-helps-cats" \
  --parameter "rerun:False" \
  --parameter "dp:1.0" \
  --partition "gh-dev" \
  --nodes=8 \
  --check_worthiness=False \
  --max-resubmissions=0 \
  --parameter "checkpoint_index:5000,7500,10000"
  # --parameter "pretrained_ckpt:/scratch/07362/gdaras/datasets/cifar-freq-single-time-blurs0_4-0.9,/scratch/07362/gdaras/datasets/cifar-freq-single-time-blurs0_6-0.9,/scratch/07362/gdaras/datasets/cifar-freq-single-time-blurs0_8-0.9" \
  # --parameter "num_generate:5239" \