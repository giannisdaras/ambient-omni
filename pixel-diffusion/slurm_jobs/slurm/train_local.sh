python -m torch.distributed.run --standalone train.py \
            --outdir=tmp \
            --data=/scratch/07362/gdaras/datasets/afhq-cats-help-multicrops \
            --corruption_probability=0.0 \
            --dataset_keep_percentage=1.0 \
            --precond=edm \
            --batch=16 \
            --workers=1 \
            --expr_id=test_training_with_annotations \
            --keep_schedule=True