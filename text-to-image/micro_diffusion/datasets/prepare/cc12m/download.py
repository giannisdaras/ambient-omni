import os
import shutil
import argparse
import subprocess
import numpy as np

from glob import iglob
from multiprocessing import Pool
from torchvision import transforms
from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError


"""Download Conceptual-Captions-12M dataset.

Example usage:
    python download.py --datadir ./cc12m/wds --valid_ids 0 1 --num_proc 2
"""


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Download Conceptual-Captions-12M dataset.'
    )
    parser.add_argument(
        '--datadir',
        type=str,
        default='./cc12m/wds',
        help='Directory to store wds data.'
    )
    parser.add_argument(
        '--valid_ids',
        type=int,
        nargs='+',
        default=list(np.arange(2176)),
        help='List of valid image IDs (default is 0 to 2176).'
    )
    parser.add_argument(
        '--num_proc',
        type=int,
        default=8,
        help='Number of parallel processes for downloading images.'
    )
    return parser.parse_args()


def download_shard(datadir: str, idx: int) -> tuple[int, bool]:
    """Downloads a single shard from HuggingFace Hub.
    
    Returns:
        tuple: (shard_idx, success_status)
    """
    try:
        hf_hub_download(
            repo_id="pixparse/cc12m-wds",
            repo_type="dataset", 
            filename=f'cc12m-train-{idx:>04}.tar',
            local_dir=datadir,
            local_dir_use_symlinks=False
        )
        return idx, True
    except Exception as e:
        print(f"Failed to download shard {idx}: {str(e)}")
        return idx, False


def main():
    args = parse_arguments()
    
    os.makedirs(args.datadir, exist_ok=True)
    
    try:
        hf_hub_download(
            repo_id="pixparse/cc12m-wds",
            repo_type="dataset",
            filename="_info.json",
            local_dir=args.datadir,
            local_dir_use_symlinks=False
        )
    except Exception as e:
        print(f"Failed to download _info.json: {str(e)}")
        print("Continuing with shard downloads...")
    
    # Use multiprocessing to download the wds dataset
    with Pool(processes=args.num_proc) as pool:
        results = pool.starmap(
            download_shard,
            [(args.datadir, idx) for idx in args.valid_ids]
        )
    
    # Collect failed downloads
    failed_shards = [idx for idx, success in results if not success]
    
    # Print summary
    print("\n" + "="*50)
    print(f"Download Summary:")
    print(f"Total shards attempted: {len(args.valid_ids)}")
    print(f"Successfully downloaded: {len(args.valid_ids) - len(failed_shards)}")
    print(f"Failed downloads: {len(failed_shards)}")
    
    # Write failed shards to a summary file
    summary_file = os.path.join(args.datadir, "failed_downloads.txt")
    with open(summary_file, "w") as f:
        f.write(f"Download Summary - {len(failed_shards)} Failed Shards\n")
        f.write(f"Timestamp: {subprocess.check_output('date', shell=True).decode().strip()}\n\n")
        
        if failed_shards:
            f.write("Failed shards:\n")
            for idx in failed_shards:
                f.write(f"  - Shard {idx:>04}\n")
        else:
            f.write("All shards were downloaded successfully!\n")
    
    print(f"\nSummary written to: {summary_file}")
    
    if failed_shards:
        print("\nThe following shards failed to download:")
        for idx in failed_shards:
            print(f"  - Shard {idx:>04}")
    else:
        print("\nAll shards were downloaded successfully!")
    print("="*50)


if __name__ == "__main__":
    main()