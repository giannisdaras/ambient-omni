# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Calculate quality metrics (BRISQUE and CLIP-IQA)."""

import os
import click
import torchmetrics.multimodal
import tqdm
import pickle
import numpy as np
import scipy.linalg
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist
from torch_utils import misc
from training import dataset
import generate_images
import piq
import torchvision
import hashlib
import torchmetrics

def get_hash(tensor):
    tensor_bytes = tensor.tobytes()
    return np.frombuffer(hashlib.sha256(tensor_bytes).digest(), dtype=np.int64).copy()

def load_stats(path, verbose=True):
    if verbose:
        print(f'Loading feature statistics from {path} ...')
    with dnnlib.util.open_url(path, verbose=verbose) as f:
        if path.lower().endswith('.npz'): # backwards compatibility with https://github.com/NVlabs/edm
            return {'fid': dict(np.load(f))}
        return pickle.load(f)

#----------------------------------------------------------------------------
# Metric specifications.

metric_specs = {
    # 'BRISQUE':          dnnlib.EasyDict(),
    'CLIP-IQA':    dnnlib.EasyDict(),
    'CLIP-IQA-512':    dnnlib.EasyDict(),
    'CLIP-IQA-256':    dnnlib.EasyDict(),
    'CLIP-IQA-128':    dnnlib.EasyDict(),
    'CLIP-IQA-64':    dnnlib.EasyDict(),
    'CLIP-IQA-32':    dnnlib.EasyDict(),
    # 'CLIP-IQA-16':    dnnlib.EasyDict(),
    # 'CLIP-IQA-8':    dnnlib.EasyDict(),
}
#----------------------------------------------------------------------------
# Get feature detector for the given metric.

_detector_cache = dict()

import torch

def patchify_batch(images: torch.Tensor, K: int) -> torch.Tensor:
    assert images.dim() == 4, "Input must be of shape (B, C, H, W)"
    B, C, H, W = images.shape
    assert H % K == 0 and W % K == 0, "Height and width must be divisible by K"

    # Unfold the height and width dimensions
    patches = images.unfold(2, K, K).unfold(3, K, K)  # (B, C, H//K, W//K, K, K)
    patches = patches.permute(0, 2, 3, 1, 4, 5)       # (B, H//K, W//K, C, K, K)
    patches = patches.reshape(B, -1, C, K, K)         # (B, N, C, K, K)
    return patches

def unpatchify_batch(patches: torch.Tensor) -> torch.Tensor:
    assert patches.dim() == 5, "Input must be of shape (B, N, C, K, K)"
    B, N, C, K, _ = patches.shape
    H = W = int(N**0.5)
    assert H % K == 0 and W % K == 0, "H and W must be divisible by K"
    num_patches_h = H // K
    num_patches_w = W // K
    assert N == num_patches_h * num_patches_w, "Mismatch in number of patches"

    # Reshape to (B, num_patches_h, num_patches_w, C, K, K)
    patches = patches.view(B, num_patches_h, num_patches_w, C, K, K)

    # Rearrange and combine to (B, C, H, W)
    images = patches.permute(0, 3, 1, 4, 2, 5)  # (B, C, H//K, K, W//K, K)
    images = images.reshape(B, C, H, W)
    return images

class PatchCLIPIQA():

    def __init__(self, clip_iqa, patch_size):
        self.clip_iqa = clip_iqa
        assert 512 % patch_size == 0, f'512 is not divisible by patch size ({patch_size})!'
        self.patch_size = patch_size
        self.resize_fn = torchvision.transforms.Resize((224, 224)) if patch_size < 224 else lambda x: x

    def __call__(self, x):
        with torch.no_grad():
            x = patchify_batch(x, self.patch_size) # B, N, C, K, K
            B, N, C, K, _ = x.shape
            x = torch.stack([self.clip_iqa(self.resize_fn(x[:, i])) for i in range(N)], dim=1) # B*N
            x = unpatchify_batch(x.view(B, N, 1, 1, 1))[:, 0, :, :] # B, N**0.5, N**0.5
        return x
    
    def to(self, device):
        self.clip_iqa = self.clip_iqa.to(device)
        return self


def get_detector(metric, verbose=True):
    # Lookup from cache.
    if metric in _detector_cache:
        return _detector_cache[metric]

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Construct detector.
    detector = None
    clip_iqa = piq.CLIPIQA(data_range=1.)
    if metric == 'BRISQUE':
        detector = piq.BRISQUELoss(reduction='none')
        if verbose:
            dist.print0(f'Setting up {metric}...')
    elif metric == 'CLIP-IQA':
        detector = clip_iqa
        if verbose:
            dist.print0(f'Setting up {metric}...')
    elif 'CLIP-IQA' in metric:
        detector = PatchCLIPIQA(clip_iqa, int(metric.split('-')[-1]))
        if verbose:
            dist.print0(f'Setting up {metric}...')
    else:
        raise Exception(dist.print0(f'Unknown metric {metric}...'))
    _detector_cache[metric] = detector

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()
    return detector

#----------------------------------------------------------------------------
# Load feature statistics from the given .pkl or .npz file.

def load_stats(path, verbose=True):
    if verbose:
        print(f'Loading feature statistics from {path} ...')
    with dnnlib.util.open_url(path, verbose=verbose) as f:
        if path.lower().endswith('.npz'): # backwards compatibility with https://github.com/NVlabs/edm
            return {'fid': dict(np.load(f))}
        return pickle.load(f)

#----------------------------------------------------------------------------
# Save feature statistics to the given .pkl file.

def save_stats(stats, path, verbose=True):
    if verbose:
        print(f'Saving feature statistics to {path} ...')
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(stats, f)

#----------------------------------------------------------------------------
# Calculate feature statistics for the given image batches
# in a distributed fashion. Returns an iterable that yields
# dnnlib.EasyDict(stats, images, batch_idx, num_batches)

def calculate_stats_for_iterable(
    image_iter,                         # Iterable of image batches: NCHW, uint8, 3 channels.
    metrics     = list(metric_specs.keys()), # Metrics to compute the statistics for.
    verbose     = True,                 # Enable status prints?
    dest_path   = None,                 # Where to save the statistics. None = do not save.
    device      = torch.device('cuda'), # Which compute device to use.
):
    # Initialize.
    num_batches = len(image_iter)
    detectors = [get_detector(metric, verbose=verbose).to(device) for metric in metrics]
    if verbose:
        dist.print0('Calculating feature statistics...')

    # Return an iterable over the batches.
    class StatsIterable:
        def __len__(self):
            return num_batches

        def __iter__(self):
            state = [dnnlib.EasyDict(metric=metric, detector=detector) for metric, detector in zip(metrics, detectors)]
            for s in state:
                s.quality = []
            cum_images = torch.zeros([], dtype=torch.int64, device=device)
            hash_list = []

            # Loop over batches.
            for batch_idx, images in enumerate(image_iter):
                if isinstance(images, dict) and 'images' in images: # dict(images)
                    images = images['images']
                elif isinstance(images, (tuple, list)) and len(images) == 2: # (images, labels)
                    images = images[0]
                images = torch.as_tensor(images).to(device) # / 255. # Normalize
                hash_list.extend([get_hash(x.detach().cpu().numpy() / 255.) for x in images])

                # Accumulate statistics.
                if images is not None:
                    for s in state:
                        detector_output = s.detector(images / 255.)
                        if detector_output.ndim == 3:
                            detector_output = detector_output.flatten(1).mean(1)
                        s.quality.append(detector_output.cpu())
                    cum_images += images.shape[0]

                # Output results.
                r = dnnlib.EasyDict(stats=None, images=images, batch_idx=batch_idx, num_batches=num_batches)
                r.num_images = int(cum_images.cpu())
                if batch_idx == num_batches - 1:
                    assert r.num_images >= 2
                    r.stats = dict(num_images=r.num_images)
                    for s in state:
                        quality = torch.concat(s.quality)
                        r.stats[s.metric] = quality #dict(quality=quality)
                    if dest_path is not None: # and dist.get_rank() == 0:
                        dest_path_rank = dest_path + f'_{dist.get_rank()}.pkl'
                        save_dict = dnnlib.EasyDict(hash=torch.from_numpy(np.concatenate(hash_list)), **r.stats)
                        save_stats(stats=save_dict, path=dest_path_rank, verbose=False)
                yield r

    return StatsIterable()

#----------------------------------------------------------------------------
# Calculate feature statistics for the given directory or ZIP of images
# in a distributed fashion. Returns an iterable that yields
# dnnlib.EasyDict(stats, images, batch_idx, num_batches)

def calculate_stats_for_files(
    image_path,             # Path to a directory or ZIP file containing the images.
    num_images      = None, # Number of images to use. None = all available images.
    seed            = 0,    # Random seed for selecting the images.
    max_batch_size  = 64,   # Maximum batch size.
    num_workers     = 2,    # How many subprocesses to use for data loading.
    prefetch_factor = 2,    # Number of images loaded in advance by each worker.
    verbose         = True, # Enable status prints?
    **stats_kwargs,         # Arguments for calculate_stats_for_iterable().
):
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # List images.
    if verbose:
        dist.print0(f'Loading images from {image_path} ...')
    dataset_obj = dataset.ImageFolderDataset(path=image_path, max_size=num_images, random_seed=seed)
    if num_images is not None and len(dataset_obj) < num_images:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but expected at least {num_images}')
    if len(dataset_obj) < 2:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but need at least 2 to compute statistics')

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide images into batches.
    if dist.get_rank() == 0:
        print('len(dataset_obj)', len(dataset_obj), 'max_batch_size', max_batch_size, 'world_size', dist.get_world_size())
    torch.distributed.barrier()
    num_batches = max((len(dataset_obj) - 1) // (max_batch_size * dist.get_world_size()) + 1, 1) * dist.get_world_size()
    rank_batches = np.array_split(np.arange(len(dataset_obj)), num_batches)[dist.get_rank() :: dist.get_world_size()]
    data_loader = torch.utils.data.DataLoader(dataset_obj, batch_sampler=rank_batches,
        num_workers=num_workers, prefetch_factor=(prefetch_factor if num_workers > 0 else None))

    # Return an interable for calculating the statistics.
    return calculate_stats_for_iterable(image_iter=data_loader, verbose=verbose, **stats_kwargs)

#----------------------------------------------------------------------------
# Parse a comma separated list of strings.

def parse_metric_list(s):
    metrics = s if isinstance(s, list) else s.split(',')
    for metric in metrics:
        if metric not in metric_specs:
            raise click.ClickException(f'Invalid metric "{metric}"')
    return metrics

#----------------------------------------------------------------------------
# Main command line.

@click.group()
def cmdline():
    """Calculate evaluation metrics (FID and FD_DINOv2).

    Examples:

    \b
    # Generate 50000 images using 8 GPUs and save them as out/*/*.png
    torchrun --standalone --nproc_per_node=8 generate_images.py \\
        --preset=edm2-img512-xxl-guid-fid --outdir=out --subdirs --seeds=0-49999

    \b
    # Calculate metrics for a random subset of 50000 images in out/
    python calculate_metrics.py calc --images=out \\
        --ref=https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl

    \b
    # Calculate metrics directly for a given model without saving any images
    torchrun --standalone --nproc_per_node=8 calculate_metrics.py gen \\
        --net=https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-s-2147483-0.130.pkl \\
        --ref=https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl \\
        --seed=123456789

    \b
    # Compute dataset reference statistics
    python calculate_metrics.py ref --data=datasets/my-dataset.zip \\
        --dest=fid-refs/my-dataset.pkl
    """

#----------------------------------------------------------------------------
# 'calc' subcommand.

@cmdline.command()
@click.option('--images', 'image_path',     help='Path to the images', metavar='PATH|ZIP',                  type=str, required=True)
@click.option('--ref', 'ref_path',          help='Dataset reference statistics ', metavar='PKL|NPZ|URL',    type=str, required=True)
@click.option('--metrics',                  help='List of metrics to compute', metavar='LIST',              type=parse_metric_list, default='fid,fd_dinov2', show_default=True)
@click.option('--num', 'num_images',        help='Number of images to use', metavar='INT',                  type=click.IntRange(min=2), default=50000, show_default=True)
@click.option('--seed',                     help='Random seed for selecting the images', metavar='INT',     type=int, default=0, show_default=True)
@click.option('--batch', 'max_batch_size',  help='Maximum batch size', metavar='INT',                       type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--workers', 'num_workers',   help='Subprocesses to use for data loading', metavar='INT',     type=click.IntRange(min=0), default=2, show_default=True)

def calc(ref_path, metrics, **opts):
    """Calculate metrics for a given set of images."""
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    if dist.get_rank() == 0:
        ref = load_stats(path=ref_path) # do this first, just in case it fails
    stats_iter = calculate_stats_for_files(metrics=metrics, **opts)
    for r in tqdm.tqdm(stats_iter, unit='batch', disable=(dist.get_rank() != 0)):
        pass
    # if dist.get_rank() == 0:
    #     calculate_metrics_from_stats(stats=r.stats, ref=ref, metrics=metrics)
    torch.distributed.barrier()

#----------------------------------------------------------------------------
# 'gen' subcommand.

@cmdline.command()
@click.option('--net',                      help='Network pickle filename', metavar='PATH|URL',             type=str, required=True)
@click.option('--ref', 'ref_path',          help='Dataset reference statistics ', metavar='PKL|NPZ|URL',    type=str, required=True)
@click.option('--metrics',                  help='List of metrics to compute', metavar='LIST',              type=parse_metric_list, default='fid,fd_dinov2', show_default=True)
@click.option('--num', 'num_images',        help='Number of images to generate', metavar='INT',             type=click.IntRange(min=2), default=50000, show_default=True)
@click.option('--seed',                     help='Random seed for the first image', metavar='INT',          type=int, default=0, show_default=True)
@click.option('--batch', 'max_batch_size',  help='Maximum batch size', metavar='INT',                       type=click.IntRange(min=1), default=32, show_default=True)

def gen(net, ref_path, metrics, num_images, seed, **opts):
    """Calculate metrics for a given model using default sampler settings."""
    dist.init()
    if dist.get_rank() == 0:
        ref = load_stats(path=ref_path) # do this first, just in case it fails
    image_iter = generate_images.generate_images(net=net, seeds=range(seed, seed + num_images), **opts)
    stats_iter = calculate_stats_for_iterable(image_iter, metrics=metrics)
    for r in tqdm.tqdm(stats_iter, unit='batch', disable=(dist.get_rank() != 0)):
        pass
    # if dist.get_rank() == 0:
    #     calculate_metrics_from_stats(stats=r.stats, ref=ref, metrics=metrics)
    torch.distributed.barrier()

#----------------------------------------------------------------------------
# 'ref' subcommand.

def merge_rank_statistics(all_indices, all_stats, total_size, device='cpu'):
    all_indices = torch.cat(all_indices)
    all_stats = torch.cat(all_stats)
    sorted_indices, sort_order = torch.sort(all_indices)
    merged_stats = torch.empty((total_size, *all_stats.shape[1:]), dtype=all_stats.dtype, device=device)
    merged_stats[sorted_indices] = all_stats[sort_order]
    return merged_stats

import datetime

@cmdline.command()
@click.option('--data', 'image_path',       help='Path to the dataset', metavar='PATH|ZIP',             type=str, required=True)
@click.option('--dest', 'dest_path',        help='Destination file', metavar='PKL',                     type=str, required=True)
@click.option('--metrics',                  help='List of metrics to compute', metavar='LIST',          type=parse_metric_list, default=list(metric_specs.keys()), show_default=True)
@click.option('--batch', 'max_batch_size',  help='Maximum batch size', metavar='INT',                   type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--workers', 'num_workers',   help='Subprocesses to use for data loading', metavar='INT', type=click.IntRange(min=0), default=2, show_default=True)

def ref(**opts):
    """Calculate dataset reference statistics for 'calc' and 'gen'."""
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    stats_iter = calculate_stats_for_files(**opts)
    for _r in tqdm.tqdm(stats_iter, unit='batch', disable=(dist.get_rank() != 0)):
        pass
    torch.distributed.barrier()
    if dist.get_rank() == 0:
        len_dataset_obj = len(dataset.ImageFolderDataset(path=opts['image_path'], max_size=None, random_seed=0))
        max_batch_size = opts['max_batch_size']
        world_size = dist.get_world_size()
        num_batches = max((len_dataset_obj - 1) // (max_batch_size * world_size) + 1, 1) * world_size
        rank_batches = [torch.LongTensor(np.concatenate(np.array_split(np.arange(len_dataset_obj), num_batches)[rank :: world_size])) for rank in range(world_size)]

        quality_stats = []
        for i in range(world_size):
            path = opts['dest_path'] + f'_{i}.pkl'
            with dnnlib.util.open_url(path, verbose=True) as f:
                if path.lower().endswith('.npz'): # backwards compatibility with https://github.com/NVlabs/edm
                    return {'fid': dict(np.load(f))}
                quality_stats.append(pickle.load(f))

        merged_stats = {}
        for key in quality_stats[0].keys():
            if key != 'num_images':
                merged_stats[key] = merge_rank_statistics(rank_batches, [q[key] for q in quality_stats], total_size=len_dataset_obj)
        save_stats(stats=merged_stats, path=opts['dest_path']+'.pkl', verbose=False)
    torch.distributed.barrier()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------
