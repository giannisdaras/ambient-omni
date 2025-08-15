# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Train diffusion-based generative model using the techniques described in the
paper "Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import json
import click
import torch
import dnnlib
from torch_utils import distributed as dist
from training import training_loop
import ambient_utils
import warnings
import wandb
import string
import random
import importlib
import copy
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

@click.command()

# Main options.
@click.option('--outdir',        help='Where to save the results', metavar='DIR',                   type=str, required=True)
@click.option('--data',          help='Path to the dataset', metavar='ZIP|DIR',                     type=str, required=True)
@click.option('--cond',          help='Train class-conditional model', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--arch',          help='Network architecture', metavar='ddpmpp|ncsnpp|adm',          type=click.Choice(['ddpmpp', 'ncsnpp', 'adm']), default='ddpmpp', show_default=True)
@click.option('--precond',       help='Preconditioning & loss function', metavar='edm|edmcls|edmcls-channelconv|emdcls-features|noise_level_classification',       type=click.Choice(['edm', 'edmcls', 'edmcls-channelconv', 'edmcls-features', 'edmcls-half', 'edmcls-half-zero', 'noise_level_classification']), default='edm', show_default=True)

# Hyperparameters.
@click.option('--duration',      help='Training duration', metavar='MIMG',                          type=click.FloatRange(min=0, min_open=True), default=200, show_default=True)
@click.option('--batch',         help='Total batch size', metavar='INT',                            type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--batch-gpu',     help='Limit batch size per GPU', metavar='INT',                    type=click.IntRange(min=1))
@click.option('--cbase',         help='Channel multiplier  [default: varies]', metavar='INT',       type=int)
@click.option('--cres',          help='Channels per resolution  [default: varies]', metavar='LIST', type=parse_int_list)
@click.option('--lr',            help='Learning rate', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=10e-4, show_default=True)
@click.option('--weight_decay',  help='Weight decay', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=False), default=0.0, show_default=True)
@click.option('--optimizer_name',  help='Optimizer name', metavar='adam|soap', type=click.Choice(['adam', 'soap']), default='adam', show_default=True)
@click.option('--ema',           help='EMA half-life', metavar='MIMG',                              type=click.FloatRange(min=0), default=0.5, show_default=True)
@click.option('--dropout',       help='Dropout probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.13, show_default=True)
@click.option('--augment',       help='Augment probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.12, show_default=True)
@click.option('--clip',          help='Gradient clipping', metavar='FLOAT',                      type=click.FloatRange(min=0, min_open=True), default=1.0, show_default=True)

# Performance-related.
@click.option('--fp16',          help='Enable mixed-precision training', metavar='BOOL',            type=bool, default=False, show_default=True)
@click.option('--ls',            help='Loss scaling', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--bench',         help='Enable cuDNN benchmarking', metavar='BOOL',                  type=bool, default=True, show_default=True)
@click.option('--cache',         help='Cache dataset in CPU memory', metavar='BOOL',                type=bool, default=True, show_default=True)
@click.option('--workers',       help='DataLoader worker processes', metavar='INT',                 type=click.IntRange(min=1), default=72, show_default=True)

# I/O-related.
@click.option("--expr_id", help="Experiment ID", type=str, default="test")
@click.option('--desc',          help='String to include in result dir name', metavar='STR',        type=str)
@click.option('--nosubdir',      help='Do not create a subdirectory for results',                   is_flag=True)
@click.option('--tick',          help='How often to print progress', metavar='KIMG',                type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--snap',          help='How often to save snapshots', metavar='TICKS',               type=click.IntRange(min=1), default=20, show_default=True)
@click.option('--dump',          help='How often to dump state', metavar='TICKS',                   type=click.IntRange(min=1), default=500, show_default=True)
@click.option('--seed',          help='Random seed  [default: random]', metavar='INT',              type=int)
@click.option('--transfer',      help='Transfer learning from network pickle', metavar='PKL|URL',   type=str)
@click.option('--resume',        help='Resume from previous training state', metavar='PT',          type=str)
@click.option('-n', '--dry-run', help='Print training options and exit',                            is_flag=True)


# Corruptions related
@click.option('--noise_config', help='Corruption configuration', type=str, default='identity')
@click.option("--corruption_probability", help="Controls what percentage of images should be corrupted.", type=float, default=0.5)
@click.option('--dataset_keep_percentage', help='Limit training samples.', type=float, default=1.0, show_default=True)
@click.option('--cls_epsilon', help='Epsilon for the classifier', type=float, default=0.05, show_default=True)
@click.option('--cls_ema_window', help='EMA window for the classifier', type=int, default=32, show_default=True)
@click.option("--overwrite_cls_labels_path", help="Path to the file where the labels are stored", type=str, default=None)
@click.option("--crop_size", help="Crop size", type=int, default=None)

# Sampler related
@click.option('--s_max', help='s_max for the infinite sampler', type=float, default=None, show_default=True)

def main(**kwargs):
    """Train diffusion-based generative model using the techniques described in the
    paper "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Train DDPM++ model for class-conditional CIFAR-10 using 8 GPUs
    torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs \\
        --data=datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp
    """
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    if dist.get_rank() == 0:
        wandb.init(project="ambient_syn", 
                   config=opts, name=opts.expr_id,
                   dir=opts.outdir)

    # import the dictionary corruptions_dict from the path noise_configs/training/{opts.noise_config}.py
    corruptions_dict = importlib.import_module(f"noise_configs.{opts.noise_config}").corruptions_dict

    # Initialize config dict.
    c = dnnlib.EasyDict()
    # TODO (@giannisdaras): maybe there is a better way to group those.
    c.clip = opts.clip
    c.cls_epsilon = opts.cls_epsilon
    c.cls_ema_window = opts.cls_ema_window
    c.overwrite_cls_labels_path = opts.overwrite_cls_labels_path
    c.crop_size = opts.crop_size
    c.dataset_kwargs = dnnlib.EasyDict(path=opts.data, use_labels=opts.cond, 
                                       cache=opts.cache,
                                       corruptions_dict=corruptions_dict,
                                       corruption_probability=opts.corruption_probability,
                                       only_positive=False,
                                       )
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=8)
    c.network_kwargs = dnnlib.EasyDict()
    c.loss_kwargs = dnnlib.EasyDict()
    c.sampler_kwargs = dnnlib.EasyDict()
    if opts.optimizer_name == 'adam':
        c.optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.lr, betas=[0.9,0.999], eps=1e-8, weight_decay=opts.weight_decay)
    elif opts.optimizer_name == 'soap':
        c.optimizer_kwargs = dnnlib.EasyDict(class_name='training.soap.SOAP', lr=opts.lr, weight_decay=opts.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {opts.optimizer_name}")

    # Validate dataset options.
    try:
        dataset_obj = ambient_utils.dataset.SyntheticallyCorruptedImageFolderDataset(**c.dataset_kwargs)
        dataset_name = dataset_obj.name
        c.dataset_kwargs.dataset_keep_percentage = opts.dataset_keep_percentage
        c.dataset_kwargs.resolution = dataset_obj.resolution # be explicit about dataset resolution
        c.dataset_kwargs.max_size = int(len(dataset_obj) * opts.dataset_keep_percentage)
        if opts.cond and not dataset_obj.has_labels:
            raise click.ClickException('--cond=True requires labels specified in dataset.json')
        del dataset_obj # conserve memory
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

    # Network architecture.
    if opts.arch == 'ddpmpp':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=128, channel_mult=[2,2,2])
    elif opts.arch == 'ncsnpp':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='fourier', encoder_type='residual', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=2, resample_filter=[1,3,3,1], model_channels=128, channel_mult=[2,2,2])
    else:
        assert opts.arch == 'adm'
        c.network_kwargs.update(model_type='DhariwalUNet', model_channels=192, channel_mult=[1,2,3,4])


    if opts.precond == "edm":
        c.network_kwargs.class_name = 'training.networks.EDMPrecond'
        c.loss_kwargs.class_name = 'training.loss.AmbientEDMLoss'
        c.sampler_kwargs.s_max = opts.s_max
    elif opts.precond == "edmcls":
        c.network_kwargs.class_name = 'training.ambient_networks.EDMPrecondCLS'
        c.loss_kwargs.class_name = 'training.loss.AmbientEDMCLSLoss'
        c.network_kwargs.num_classes = 1
    else:
        raise ValueError(f"Unknown preconditioning: {opts.precond}")


    # Network options.
    if opts.cbase is not None:
        c.network_kwargs.model_channels = opts.cbase
    if opts.cres is not None:
        c.network_kwargs.channel_mult = opts.cres
    if opts.augment:
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', p=opts.augment)
        # c.augment_kwargs.update(xflip=1e8, yflip=1, scale=0, rotate_frac=1, aniso=0, translate_frac=0)
        # c.network_kwargs.augment_dim = 4
        c.augment_kwargs.update(xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1)
        c.network_kwargs.augment_dim = 9
    c.network_kwargs.update(dropout=opts.dropout, use_fp16=opts.fp16)

    # Training options.
    c.total_kimg = max(int(opts.duration * 1000), 1)
    c.ema_halflife_kimg = int(opts.ema * 1000)
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(loss_scaling=opts.ls, cudnn_benchmark=opts.bench)
    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=opts.dump)

    # Random seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    # Transfer learning and resume.
    if opts.transfer is not None:
        if opts.resume is not None:
            raise click.ClickException('--transfer and --resume cannot be specified at the same time')
        c.resume_pkl = opts.transfer
        c.ema_rampup_ratio = None
    elif opts.resume is not None:
        match = re.fullmatch(r'training-state-(\d+).pt', os.path.basename(opts.resume))
        if not match or not os.path.isfile(opts.resume):
            raise click.ClickException('--resume must point to training-state-*.pt from a previous training run')
        c.resume_pkl = os.path.join(os.path.dirname(opts.resume), f'network-snapshot-{match.group(1)}.pkl')
        c.resume_kimg = int(match.group(1))
        c.resume_state_dump = opts.resume

    # Description string.
    cond_str = 'cond' if c.dataset_kwargs.use_labels else 'uncond'
    dtype_str = 'fp16' if c.network_kwargs.use_fp16 else 'fp32'
    desc = f'{dataset_name:s}-{cond_str:s}-{opts.arch:s}-{opts.precond:s}-gpus{dist.get_world_size():d}-batch{c.batch_size:d}-{dtype_str:s}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Pick output directory.
    if dist.get_rank() != 0:
        c.run_dir = None
    elif opts.nosubdir:
        c.run_dir = opts.outdir
    else:
        prev_run_dirs = []
        if os.path.isdir(opts.outdir):
            prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        # add a random string of length 5 to run_dir
        random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
        c.run_dir = os.path.join(opts.outdir, f'{cur_run_id:05d}-{desc}-{random_string}')
        assert not os.path.exists(c.run_dir)

    # Print options.
    dist.print0()
    dist.print0('Training options:')
    c_copy = copy.deepcopy(c)  # this is done to avoid serialization issues with the functions in corruptions_dict
    c_copy.dataset_kwargs.corruptions_dict = None
    c_copy.dataset_kwargs.noise_config = opts.noise_config
    dist.print0(json.dumps(c_copy, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Dataset path:            {c.dataset_kwargs.path}')
    dist.print0(f'Class-conditional:       {c.dataset_kwargs.use_labels}')
    dist.print0(f'Network architecture:    {opts.arch}')
    dist.print0(f'Preconditioning & loss:  {opts.precond}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'Mixed-precision:         {c.network_kwargs.use_fp16}')
    dist.print0()

    # Dry run?
    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c_copy, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    del c.dataset_kwargs.dataset_keep_percentage
    # Train.
    training_loop.training_loop(**c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
