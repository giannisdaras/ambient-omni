slurmify submit-parametric-array \
  --account CGAI24022 \
  --job-name train_afhq \
  --script-path slurm_jobs/train_net.py \
  --time-limit 16:00:00 \
  --parameter "dataset_path:/scratch/07362/gdaras/datasets/afhq-blurs0_6-blurs0_6-0.9-Ms8Tu-010035,/scratch/07362/gdaras/datasets/afhq-blurs0_8-blurs0_8-0.9-fYCoL-010035,/scratch/07362/gdaras/datasets/afhq-blurs1-blurs1-0.9-09DnR-010035/" \
  --parameter "dp:1.0" \
  --parameter "keep_schedule:True" \
  --partition "gh" \
  --nodes=16 \
  --check_worthiness=False \
  --max-resubmissions=0

