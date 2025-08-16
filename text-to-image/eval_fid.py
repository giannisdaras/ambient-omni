#!/usr/bin/env python3
import argparse
import sys
from cleanfid import fid

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute FID score between two image directories using cleanfid"
    )
    parser.add_argument(
        "--dir1",
        help="Path to the first directory of images",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dir2",
        help="Path to the second directory of images",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for feature extraction (default: 50)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run on (e.g. 'cuda' or 'cpu'; default: cuda)",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    try:
        score = fid.compute_fid(
            args.dir1,
            args.dir2,
            batch_size=args.batch_size,
            device=args.device,
        )
    except Exception as e:
        print(f"Error computing FID: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"FID score between\n  {args.dir1}\nand\n  {args.dir2}\nis: {score:.6f}")

if __name__ == "__main__":
    main()