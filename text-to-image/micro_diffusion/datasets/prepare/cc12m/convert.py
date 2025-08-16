import os
import glob
import shutil
import tarfile
import numpy as np
from PIL import Image, UnidentifiedImageError
from argparse import ArgumentParser
from multiprocessing import Pool, current_process
from streaming.base import MDSWriter
from streaming.base.util import merge_index
from torchvision import transforms
from tqdm import tqdm
from typing import List, Generator, Tuple


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        '--wds_dir',
        type=str,
        required=True,
        help='Path to local dir with wds download of cc12m dataset'
    )
    parser.add_argument(
        '--local_mds_dir',
        type=str,
        default='',
        help='Directory to store mds shards.'
    )
    parser.add_argument(
        '--max_image_size',
        type=int,
        default=512,
        help='If min(h, w) > max_image_size, then downsize the smaller edge to max_image size.'
    )
    parser.add_argument(
        '--min_image_size',
        type=int,
        default=256,
        help='Skip image if any side is smaller than min_image_size.'
    )
    parser.add_argument('--num_proc', type=int, default=16)
    return parser.parse_args()


def current_process_index() -> int:
    # by default it starts from 1
    p = current_process()
    return p._identity[0] - 1


def read_tar(path: str, path_out: str) -> Generator[Tuple[Image.Image, str], None, None]:
    # Ensure the temp directory for this specific tar is clean before extraction
    if os.path.exists(path_out):
        shutil.rmtree(path_out)
    os.makedirs(path_out, exist_ok=True)

    try:
        with tarfile.open(path, 'r') as tar:
            tar.extractall(path_out)

        txts = sorted(glob.glob(os.path.join(path_out, '*txt')))
        print(f"Found {len(txts)} images in tar file {os.path.basename(path)} for process {current_process_index()}")

        for t in txts:
            try:
                with open(t, 'r') as ct:
                    cap = ct.read()
                # assuming all files are in jpg
                img_path = t.replace('.txt', '.jpg')
                if not os.path.exists(img_path):
                    print(f"Warning: Image file not found: {img_path}")
                    continue
                img = Image.open(img_path)
                yield img, cap
            # Catch more specific exceptions if possible
            except FileNotFoundError:
                 print(f"Warning: Missing file for text: {t}")
            except UnidentifiedImageError:
                 print(f"Warning: Could not identify image file corresponding to {t}")
            except Exception as e:
                print(f"Error processing file {t}: {e}")

        print(f"Done reading the tar file {os.path.basename(path)} for process {current_process_index()}")

    finally:
        # Safely remove the temporary directory for this tar file
        # Use a simple rmtree first, add retry logic if needed later
        if os.path.exists(path_out):
             try:
                 shutil.rmtree(path_out)
                 print(f"Cleaned up temp dir: {path_out}")
             except OSError as e:
                 print(f"Warning: Failed to remove {path_out}: {e}")


def write_tar(tars: List[str], args: ArgumentParser):
    columns = {
        'width': 'int32',
        'height': 'int32',
        'jpg': 'jpeg',
        'caption': 'str'
    }

    process_idx = current_process_index()
    save_dir = os.path.join(args.local_mds_dir, str(process_idx))
    os.makedirs(save_dir, exist_ok=True)

    # create a writer per process
    writer = MDSWriter(
        out=save_dir,
        columns=columns,
        compression=None,
        size_limit=256 * (2**20),
        max_workers=72
    )

    downsize = transforms.Resize(
        args.max_image_size,
        antialias=True,
        interpolation=transforms.InterpolationMode.BICUBIC
    )

    # Base temporary directory for this process
    process_temp_base = os.path.join(save_dir, f'temp_{process_idx}')
    os.makedirs(process_temp_base, exist_ok=True)

    for tar in tqdm(tars, desc=f"Process {process_idx}"):
        # Create a unique temp directory for *each* tar file
        tar_filename = os.path.basename(tar).replace('.tar', '')
        unique_temp_dir = os.path.join(process_temp_base, tar_filename + "_temp")

        rejected, written_this_tar = 0, 0
        try:
            # Pass the unique temp dir to read_tar
            for img, cap in read_tar(tar, unique_temp_dir):
                # Initial w, h before potential downsizing
                orig_w, orig_h = img.size

                try:
                    # Filter based on original size *before* downsizing
                    if min(orig_w, orig_h) < args.min_image_size:
                        rejected += 1
                        continue

                    processed_img = img
                    # Downsize if necessary
                    if min(orig_w, orig_h) > args.max_image_size:
                        processed_img = downsize(img)

                    mds_sample = {
                        'jpg': processed_img,
                        'caption': cap,
                        'width': orig_w,
                        'height': orig_h,
                    }
                    writer.write(mds_sample)
                    written_this_tar += 1

                except (UnidentifiedImageError, OSError) as e:
                    print(f"Error processing image from {tar}: {e}")
                    rejected += 1

            print(f"Process {process_idx}: Tar {os.path.basename(tar)} - Wrote: {written_this_tar}, Rejected/Skipped: {rejected}")

        except Exception as e:
            print(f"Error processing tar file {tar}: {e}")
        # Cleanup within read_tar handles the unique_temp_dir

    writer.finish()
    # Optionally clean up the process_temp_base dir if it's empty or no longer needed
    try:
        # Check if base temp dir exists AND is empty before removing
        if os.path.exists(process_temp_base) and not os.listdir(process_temp_base):
            os.rmdir(process_temp_base)
            print(f"Cleaned up base temp dir: {process_temp_base}")
    except OSError as e:
         print(f"Warning: Could not remove base temp dir {process_temp_base}: {e}")


def main():
    args = parse_arguments()
    print(os.path.join(args.wds_dir, '*tar'))
    tars = glob.glob(os.path.join(args.wds_dir, '*tar'))
    print(f"Total {len(tars)} tar files found in cc12m wds dataset path!")

    tars_split = np.array_split(tars, args.num_proc)
    
    with Pool(processes=args.num_proc) as pool:
        pool.starmap(write_tar, [(ts, args) for ts in tars_split])
    
    shards_metadata = [
        os.path.join(args.local_mds_dir, str(i), 'index.json')
        for i in range(args.num_proc)
    ]
    merge_index(shards_metadata, out=args.local_mds_dir, keep_local=True)


if __name__ == '__main__':
    main()