#!/usr/bin/env bash

DIR1= # Path to generated images
DIR2=./datadir/coco/raw/val2014
BATCH_SIZE=64

# --- Run the Python script ---
python3 compute_fid.py \
  "$DIR1" \
  "$DIR2" \
  --batch-size "$BATCH_SIZE"

# --- End ---
