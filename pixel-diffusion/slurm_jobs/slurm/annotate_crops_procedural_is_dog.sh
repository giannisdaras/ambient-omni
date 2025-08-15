# classifier gives probability of being procedural
slurmify submit-parametric-array \
  --account CGAI24022 \
  --job-name annotation \
  --script-path slurm_jobs/annotate_crops.py \
  --time-limit 01:00:00 \
  --parameter "dataset:afhq" \
  --parameter "dataset_path:/scratch/07362/gdaras/datasets/afhqv2-64x64-partitioned/1/" \
  --parameter "annotated_dataset_path:/scratch/07362/gdaras/datasets/afhq-procedural-helps-dogs/" \
  --parameter "corruption_probability:0.0" \
  --parameter "checkpoint_paths:/scratch/07362/gdaras/ambient-syn-runs/00088-dogs-and-synthetic-uncond-ddpmpp-edmcls-gpus8-batch512-fp32-U6A74/network-snapshot-017060.pkl\\,/scratch/07362/gdaras/ambient-syn-runs/00089-dogs-and-synthetic-uncond-ddpmpp-edmcls-gpus8-batch512-fp32-SxT5V/network-snapshot-014049.pkl\\,/scratch/07362/gdaras/ambient-syn-runs/00091-dogs-and-synthetic-uncond-ddpmpp-edmcls-gpus8-batch512-fp32-RledY/network-snapshot-008028.pkl\\,/scratch/07362/gdaras/ambient-syn-runs/00090-dogs-and-synthetic-uncond-ddpmpp-edmcls-gpus8-batch512-fp32-H2yLL/network-snapshot-010035.pkl" \
  --parameter "save_only_clean:False" \
  --parameter "flip_probs:True" \
  --partition "gh" \
  --nodes=8 \
  --check_worthiness=False \
  --max-resubmissions=0

