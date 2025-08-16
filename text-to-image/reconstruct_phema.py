# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Perform post-hoc EMA reconstruction."""

import os
import re
import copy
import warnings
import click
import tqdm
import numpy as np
import torch
from ambient_utils.utils import EasyDict
from ambient_utils.url import is_url
# import dnnlib
import sys
import os

sys.path.append("../edm2/training")
from phema import solve_posthoc_coefficients

warnings.filterwarnings('ignore', 'You are using `torch.load` with `weights_only=False`')

#----------------------------------------------------------------------------
# Construct the full path of a network file.

def pt_path(dir, prefix, nimg, std):
    name = prefix + f'-{nimg//1000:07d}-{std:.3f}.pt'
    if dir is None:
        return None
    if is_url(dir):
        return f'{dir}/{name}'
    return os.path.join(dir, name)

#----------------------------------------------------------------------------
# Deduce nimg based on kimg (= nimg//1000).

def kimg_to_nimg(kimg):
    nimg = (kimg * 1000 + 999) // 1024 * 1024
    assert nimg // 1000 == kimg
    return nimg

#----------------------------------------------------------------------------
# List input pt files for post-hoc EMA reconstruction.
# Returns a list of EasyDict(path, nimg, std).

def list_input_files(
    in_dir,             # Directory containing the input files.
    in_prefix   = None, # Input filename prefix. None = anything goes.
    in_std      = None, # Relative standard deviations of the input files. None = anything goes.
):
    if not os.path.isdir(in_dir):
        raise click.ClickException('Input directory does not exist')
    in_std = set(in_std) if in_std is not None else None

    files = []
    with os.scandir(in_dir) as it:
        for e in it:
            m = re.fullmatch(r'(.*)-(\d+)-(\d+\.\d+)\.pt', e.name)
            if not m or not e.is_file():
                continue
            prefix = m.group(1)
            nimg = kimg_to_nimg(int(m.group(2)))
            std = float(m.group(3))
            if in_prefix is not None and prefix != in_prefix:
                continue
            if in_std is not None and std not in in_std:
                continue
            files.append(EasyDict(path=e.path, nimg=nimg, std=std))
    files = sorted(files, key=lambda file: (file.nimg, file.std))
    return files

#----------------------------------------------------------------------------
# Perform post-hoc EMA reconstruction.
# Returns an iterable that yields EasyDict(out, step_idx, num_steps),
# where 'out' is a list of EasyDict(net, nimg, std, model_data, pt_path)

def reconstruct_phema(
    in_files,                   # List of input files, expressed as EasyDict(path, nimg, std).
    out_std,                    # List of relative standard deviations to reconstruct.
    out_nimg        = None,     # Training time of the snapshot to reconstruct. None = highest input time.
    out_dir         = None,     # Where to save the reconstructed network files. None = do not save.
    out_prefix      = 'phema',  # Output filename prefix.
    skip_existing   = False,    # Skip output files that already exist?
    max_batch_size  = 8,        # Maximum simultaneous reconstructions
    verbose         = True,     # Enable status prints?
):
    # Validate input files.
    if out_nimg is None:
        out_nimg = max((file.nimg for file in in_files), default=0)
    elif not any(out_nimg == file.nimg for file in in_files):
        raise click.ClickException('Reconstruction time must match one of the input files')
    in_files = [file for file in in_files if 0 < file.nimg <= out_nimg]
    if len(in_files) == 0:
        raise click.ClickException('No valid input files found')
    in_nimg = [file.nimg for file in in_files]
    in_std = [file.std for file in in_files]
    if verbose:
        print(f'Loading {len(in_files)} input files...')
        for file in in_files:
            print('    ' + file.path)

    # Determine output files.
    out_std = [out_std] if isinstance(out_std, float) else sorted(set(out_std))
    if skip_existing and out_dir is not None:
        out_std = [std for std in out_std if not os.path.isfile(pt_path(out_dir, out_prefix, out_nimg, std))]
    num_batches = (len(out_std) - 1) // max_batch_size + 1
    out_std_batches = np.array_split(out_std, num_batches)
    if verbose:
        print(f'Reconstructing {len(out_std)} output files in {num_batches} batches...')
        for i, batch in enumerate(out_std_batches):
            for std in batch:
                print(f'    batch {i}: ', end='')
                print(pt_path(out_dir, out_prefix, out_nimg, std) if out_dir is not None else pt_path('', '<yield>', out_nimg, std))

    # Return an iterable over the reconstruction steps.
    class ReconstructionIterable:
        def __len__(self):
            return num_batches * len(in_files)

        def __iter__(self):
            # Loop over batches.
            r = EasyDict(step_idx=0, num_steps=len(self))
            for out_std_batch in out_std_batches:
                coefs = solve_posthoc_coefficients(in_nimg, in_std, out_nimg, out_std_batch)
                out = [EasyDict(net=None, nimg=out_nimg, std=std) for std in out_std_batch]
                r.out = []

                # Loop over input files.
                for i in range(len(in_files)):
                    try:
                        in_model_data = torch.load(in_files[i].path, weights_only=False)
                        if 'ema' in in_model_data:
                            in_net = in_model_data['ema']
                            found_key = 'ema'
                        elif 'state'in in_model_data:
                            in_net = in_model_data['state']
                            found_key = 'state'
                        else:
                            raise ValueError(f"Unknown model data format in {in_files[i].path}")
                    except Exception as e:
                        print(f"Error loading file {in_files[i].path}: {e}")
                        raise

                    # Accumulate weights for each output file.
                    for j in range(len(out)):
                        if out[j].net is None:
                            out[j].model_data = copy.deepcopy(in_model_data)
                            out[j].net = out[j].model_data[found_key]
                            # Initialize weights to zero
                            for key, tensor in out[j].net.items():
                                if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                                    out[j].net[key] = torch.zeros_like(tensor)
                        # Update weights with coefficients
                        for key, tensor in in_net.items():
                            if isinstance(tensor, torch.Tensor):
                                if tensor.requires_grad:
                                    # Parameters get weighted sum
                                    out[j].net[key] += tensor * coefs[i, j]
                                else:
                                    # Buffers get copied directly
                                    out[j].net[key] = tensor.clone()

                    # Finalize outputs.
                    if i == len(in_files) - 1:
                        for j in range(len(out)):
                            out[j].pt_path = pt_path(out_dir, out_prefix, out_nimg, out[j].std)
                            if out[j].pt_path is not None:
                                os.makedirs(out_dir, exist_ok=True)
                                torch.save(out[j].model_data, out[j].pt_path)
                        r.out = out

                    # Yield results.
                    del in_model_data, in_net # conserve memory
                    yield r
                    r.step_idx += 1

    return ReconstructionIterable()

#----------------------------------------------------------------------------
# Parse a comma separated list of relative standard deviations.
# The special token '...' interpreted as an evenly spaced interval.
# Example: '0.01,0.02,...,0.05' returns [0.01, 0.02, 0.03, 0.04, 0.05]

def parse_std_list(s):
    if isinstance(s, list):
        return s

    # Parse raw values.
    raw = [None if v == '...' else float(v) for v in s.split(',')]

    # Fill in '...' tokens.
    out = []
    for i, v in enumerate(raw):
        if v is not None:
            out.append(v)
            continue
        if i - 2 < 0 or raw[i - 2] is None or raw[i - 1] is None:
            raise click.ClickException("'...' must be preceded by at least two floats")
        if i + 1 >= len(raw) or raw[i + 1] is None:
            raise click.ClickException("'...' must be followed by at least one float")
        if raw[i - 2] == raw[i - 1]:
            raise click.ClickException("The floats preceding '...' must not be equal")
        approx_num = (raw[i + 1] - raw[i - 1]) / (raw[i - 1] - raw[i - 2]) - 1
        num = round(approx_num)
        if num <= 0:
            raise click.ClickException("'...' must correspond to a non-empty interval")
        if abs(num - approx_num) > 1e-4:
            raise click.ClickException("'...' must correspond to an evenly spaced interval")
        for j in range(num):
            out.append(raw[i - 1] + (raw[i - 1] - raw[i - 2]) * (j + 1))

    # Validate.
    out = sorted(set(out))
    if not all(0.000 < v < 0.289 for v in out):
        raise click.ClickException('Relative standard deviation must be positive and less than 0.289')
    return out

#----------------------------------------------------------------------------
# Command line interface.

@click.command()
@click.option('--indir', 'in_dir',          help='Directory containing the input files', metavar='DIR',           type=str, required=True)
@click.option('--inprefix', 'in_prefix',    help='Filter inputs based on filename prefix', metavar='STR',           type=str, default=None)
@click.option('--instd', 'in_std',          help='Filter inputs based on standard deviations', metavar='LIST',      type=parse_std_list, default=None)

@click.option('--outdir', 'out_dir',        help='Where to save the reconstructed network files', metavar='DIR',  type=str, required=True)
@click.option('--outprefix', 'out_prefix',  help='Output filename prefix', metavar='STR',                           type=str, default='phema', show_default=True)
@click.option('--outstd', 'out_std',        help='List of desired relative standard deviations', metavar='LIST',    type=parse_std_list, required=True)
@click.option('--outkimg', 'out_kimg',      help='Training time of the snapshot to reconstruct', metavar='KIMG',    type=click.IntRange(min=1), default=None)

@click.option('--skip', 'skip_existing',    help='Skip output files that already exist',                            is_flag=True)
@click.option('--batch', 'max_batch_size',  help='Maximum simultaneous reconstructions', metavar='INT',             type=click.IntRange(min=1), default=8, show_default=True)

def cmdline(in_dir, in_prefix, in_std, out_kimg, **opts):
    """Perform post-hoc EMA reconstruction.

    Examples:

    \b
    # Download raw snapshots for the pre-trained edm2-img512-xs model
    rclone copy --progress --http-url https://nvlabs-fi-cdn.nvidia.com/edm2 \\
        :http:raw-snapshots/edm2-img512-xs/ raw-snapshots/edm2-img512-xs/

    \b
    # Reconstruct a new EMA profile with std=0.150
    python reconstruct_phema.py --indir=raw-snapshots/edm2-img512-xs \\
        --outdir=out --outstd=0.150

    \b
    # Reconstruct a set of 31 EMA profiles, streaming over the input data 4 times
    python reconstruct_phema.py --indir=raw-snapshots/edm2-img512-xs \\
        --outdir=out --outstd=0.010,0.015,...,0.250 --batch=8

    \b
    # Perform reconstruction for the latest snapshot of a given training run
    python reconstruct_phema.py --indir=training-runs/00000-edm2-img512-xs \\
        --outdir=out --outstd=0.150
    """
    if os.environ.get('WORLD_SIZE', '1') != '1':
        raise click.ClickException('Distributed execution is not supported')
    out_nimg = kimg_to_nimg(out_kimg) if out_kimg is not None else None
    in_files = list_input_files(in_dir=in_dir, in_prefix=in_prefix, in_std=in_std)
    # in_files = in_files[18:]  # TODO: remove this -- it is here just to make things faster
    rec_iter = reconstruct_phema(in_files=in_files, out_nimg=out_nimg, **opts)
    for _r in tqdm.tqdm(rec_iter, unit='step'):
        pass

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------