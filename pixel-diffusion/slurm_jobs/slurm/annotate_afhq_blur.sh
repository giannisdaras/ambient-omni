slurmify submit-parametric-array \
  --account CGAI24022 \
  --job-name annotate_afhq \
  --script-path slurm_jobs/annotate_cls.py \
  --time-limit 12:00:00 \
  --parameter "dataset:afhq" \
  --parameter "dataset_path:/scratch/07362/gdaras/datasets/afhqv2-64x64.zip" \
  --parameter "corruption_probability:0.9" \
  --parameter "training_noise_config:blurs0_4" \
  --parameter "inference_noise_config:blurs0_4" \
  --parameter "checkpoint_path:/scratch/07362/gdaras/ambient-syn-runs/00112-afhqv2-64x64-uncond-ddpmpp-edmcls-gpus8-batch512-fp32-2dRie/network-snapshot-030106.pkl" \
  --partition "gh" \
  --nodes=16 \
  --check_worthiness=False \
  --max-resubmissions=0


# 1: /scratch/07362/gdaras/ambient-syn-runs/00110-afhqv2-64x64-uncond-ddpmpp-edmcls-gpus8-batch512-fp32-09DnR/network-snapshot-010035.pkl
# 0.8: /scratch/07362/gdaras/ambient-syn-runs/00111-afhqv2-64x64-uncond-ddpmpp-edmcls-gpus8-batch512-fp32-fYCoL/network-snapshot-010035.pkl
# 0.6: /scratch/07362/gdaras/ambient-syn-runs/00113-afhqv2-64x64-uncond-ddpmpp-edmcls-gpus8-batch512-fp32-Ms8Tu/network-snapshot-010035.pkl
# 0.4: /scratch/07362/gdaras/ambient-syn-runs/00112-afhqv2-64x64-uncond-ddpmpp-edmcls-gpus8-batch512-fp32-2dRie/network-snapshot-010035.pkl