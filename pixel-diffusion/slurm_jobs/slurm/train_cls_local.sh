python train.py \
  --outdir=tmp \
  --data=/scratch/07362/gdaras/datasets/dogs-and-synthetic/ \
  --precond=edmcls \
  --noise_config=identity \
  --corruption_probability=0.0 \
  --dataset_keep_percentage=1.0 \
  --overwrite_cls_labels_path=/scratch/07362/gdaras/datasets/dogs-and-synthetic/labels.jsonl \
  --crop_size=4 \
  --expr_id=test