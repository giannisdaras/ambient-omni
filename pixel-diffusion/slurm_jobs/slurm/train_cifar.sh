slurmify submit-parametric-array \
  --account CGAI24022 \
  --job-name train_cifar \
  --script-path slurm_jobs/train_net.py \
  --time-limit 16:00:00 \
  --parameter "dataset_path:/scratch/07362/gdaras/datasets/cifar-fixed-sigma-0.1-0.0-blurs0_4-0.9/" \
  --parameter "dp:1.0" \
  --parameter "keep_schedule:True" \
  --partition "gh" \
  --nodes=8 \
  --check_worthiness=False \
  --max-resubmissions=0 \
  # --parameter "checkpoint:/scratch/07362/gdaras/ambient-syn-runs/00011-afhq-dogs-uncond-ddpmpp-edm-gpus16-batch512-fp32-GQvJa/training-state-025088.pt" \
  # --parameter "dataset_path:/scratch/07362/gdaras/datasets/cifar-fixed-sigma-0.2-0.0-blurs0_4-0.9/,/scratch/07362/gdaras/datasets/cifar-fixed-sigma-0.4-0.0-blurs0_4-0.9/,/scratch/07362/gdaras/datasets/cifar-fixed-sigma-0.6-0.0-blurs0_4-0.9/,/scratch/07362/gdaras/datasets/cifar-fixed-sigma-0.8-0.0-blurs0_4-0.9/" \
