# source slurm_jobs/slurm/prepare_cls_dataset.sh /scratch/07362/gdaras/datasets/afhq-cats-and-wildlife /scratch/07362/gdaras/datasets/afhqv2-64x64-partitioned/0/:100 /scratch/07362/gdaras/datasets/afhqv2-64x64-part2/:100
slurmify submit-parametric-array \
  --account CGAI24022 \
  --job-name train_cls_cats_and_wildlife \
  --script-path slurm_jobs/train_cls.py \
  --time-limit 16:00:00 \
  --parameter "dataset:afhq" \
  --parameter "dataset_path:/scratch/07362/gdaras/datasets/dogs-and-synthetic" \
  --parameter "noise_config:identity" \
  --parameter "corruption_probability:0." \
  --parameter "all_pairs:False" \
  --parameter "dp:1.0" \
  --partition "gh" \
  --nodes=8 \
  --check_worthiness=False \
  --max-resubmissions=0 \
  --parameter "overwrite_cls_labels_path:/scratch/07362/gdaras/datasets/dogs-and-synthetic/labels.jsonl" \
  --parameter "crop_size:4,8,16,24" 