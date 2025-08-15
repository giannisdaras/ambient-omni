slurmify submit-parametric-array \
  --account CGAI24022 \
  --job-name eval_afhq \
  --script-path slurm_jobs/eval_net.py \
  --time-limit 00:15:00 \
  --parameter "dataset:afhq" \
  --parameter "noise_config:blurs0_8" \
  --parameter "dataset_path:/scratch/07362/gdaras/datasets/afhq-blurs0_8-blurs0_8-0.9-fYCoL-010035/" \
  --parameter "rerun:False" \
  --parameter "dp:1.0" \
  --partition "gh" \
  --nodes=8 \
  --check_worthiness=False \
  --parameter "checkpoint_index:5000,6000,7000,8000,9000,10000,11000,12000,13000,14000" \
  --max-resubmissions=0 \
  --parameter "num_generate:15803"

