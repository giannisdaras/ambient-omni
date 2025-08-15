slurmify submit-parametric-array \
  --account CGAI24022 \
  --job-name eval_afhq \
  --script-path slurm_jobs/eval_net.py \
  --time-limit 00:15:00 \
  --parameter "dataset:afhq" \
  --parameter "noise_config:identity" \
  --parameter "dataset_path:/scratch/07362/gdaras/datasets/afhq-fixed-sigma-0.0-0.0-blurs1-0.9/" \
  --parameter "rerun:False" \
  --parameter "dp:1.0" \
  --partition "gh" \
  --nodes=8 \
  --check_worthiness=False \
  --parameter "checkpoint_index:10000,15000,20000,25000,30000,35000,40000,45000,50000,55000,60000,65000" \
  --max-resubmissions=0 \
  --parameter "num_generate:15803"

#   --parameter "dataset_path:/scratch/07362/gdaras/datasets/afhq-fixed-sigma-0.0-0.0-blurs0_4-0.9/,/scratch/07362/gdaras/datasets/afhq-fixed-sigma-0.0-0.0-blurs0_6-0.9/,/scratch/07362/gdaras/datasets/afhq-fixed-sigma-0.0-0.0-blurs0_8-0.9/,/scratch/07362/gdaras/datasets/afhq-fixed-sigma-0.0-0.0-blurs1-0.9/" \
