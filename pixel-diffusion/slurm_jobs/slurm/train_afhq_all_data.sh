slurmify submit-parametric-array \
  --account CGAI24022 \
  --job-name train_afhq \
  --script-path slurm_jobs/train_net.py \
  --time-limit 12:00:00 \
  --parameter "dataset_path:/scratch/07362/gdaras/datasets/afhq-fixed-sigma-0.0-0.0-identity-0.9-only-clean" \
  --parameter "dp:1.0" \
  --parameter "corruption_probability:0.0" \
  --parameter "keep_schedule:True" \
  --partition "gh" \
  --nodes=16 \
  --check_worthiness=False \
  --max-resubmissions=0

    # --parameter "dataset_path:/scratch/07362/gdaras/datasets/afhq-fixed-sigma-0.0-0.0-blurs0_4-0.9/,/scratch/07362/gdaras/datasets/afhq-fixed-sigma-0.0-0.0-blurs0_6-0.9/,/scratch/07362/gdaras/datasets/afhq-fixed-sigma-0.0-0.0-blurs0_8-0.9/,/scratch/07362/gdaras/datasets/afhq-fixed-sigma-0.0-0.0-blurs1-0.9/" \

