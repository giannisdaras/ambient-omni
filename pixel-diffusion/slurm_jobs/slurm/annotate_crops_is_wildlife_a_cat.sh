# classifier gives probability of being wildlife
slurmify submit-parametric-array \
  --account CGAI24022 \
  --job-name annotation \
  --script-path slurm_jobs/annotate_crops.py \
  --time-limit 01:00:00 \
  --parameter "dataset:afhq" \
  --parameter "dataset_path:/scratch/07362/gdaras/datasets/afhqv2-64x64-part2/" \
  --parameter "annotated_dataset_path:/scratch/07362/gdaras/datasets/afhq-wildlife-helps-cats/" \
  --parameter "corruption_probability:0.0" \
  --parameter "checkpoint_paths:/scratch/07362/gdaras/ambient-syn-runs/00084-afhq-cats-and-wildlife-uncond-ddpmpp-edmcls-gpus8-batch512-fp32-xgujP/network-snapshot-020070.pkl\\,/scratch/07362/gdaras/ambient-syn-runs/00085-afhq-cats-and-wildlife-uncond-ddpmpp-edmcls-gpus8-batch512-fp32-rUXRu/network-snapshot-020070.pkl\\,/scratch/07362/gdaras/ambient-syn-runs/00086-afhq-cats-and-wildlife-uncond-ddpmpp-edmcls-gpus8-batch512-fp32-hQ0l1/network-snapshot-020070.pkl\\,/scratch/07362/gdaras/ambient-syn-runs/00087-afhq-cats-and-wildlife-uncond-ddpmpp-edmcls-gpus8-batch512-fp32-GJfrP/network-snapshot-020070.pkl" \
  --parameter "save_only_clean:False" \
  --parameter "flip_probs:True" \
  --partition "gh" \
  --nodes=8 \
  --check_worthiness=False \
  --max-resubmissions=0

