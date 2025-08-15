# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import psutil
import numpy as np
import torch
import torch.nn as nn
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
import ambient_utils
import wandb
from ambient_utils.classifier import analyze_classifier_trajectory
from collections import defaultdict
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import pandas as pd
#----------------------------------------------------------------------------

def sample_t(current_sigma, loc, scale):
    numpy_cutoff = current_sigma.squeeze().cpu().numpy()
    numpy_cutoff = (np.log(numpy_cutoff) - loc) / scale
    sampled = truncnorm.rvs(numpy_cutoff, np.inf, loc=loc, scale=scale)
    return torch.tensor(sampled, device=current_sigma.device).exp().view_as(current_sigma)


def apply_ema(data, window=3):
    return pd.Series(data).ewm(span=window, adjust=False).mean().values


def training_loop(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for training set.
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_kwargs      = {},       # Options for model and preconditioning.
    loss_kwargs         = {},       # Options for loss function.
    optimizer_kwargs    = {},       # Options for optimizer.
    augment_kwargs      = None,     # Options for augmentation pipeline, None = disable.
    seed                = 0,        # Global random seed.
    batch_size          = 512,      # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_kimg          = 200000,   # Training duration, measured in thousands of training images.
    ema_halflife_kimg   = 500,      # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio    = 0.05,     # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg      = 10000,    # Learning rate ramp-up duration.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick       = 50,       # Interval of progress prints.
    snapshot_ticks      = 50,       # How often to save network snapshots, None = disable.
    state_dump_ticks    = 500,      # How often to dump training state, None = disable.
    resume_pkl          = None,     # Start from the given network snapshot, None = random initialization.
    resume_state_dump   = None,     # Start from the given training state, None = reset training state.
    resume_kimg         = 0,        # Start from the given training progress.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
    clip                = 1.0,
    cls_epsilon         = 0.05,
    cls_ema_window      = 32,
    overwrite_cls_labels_path = None, # if not None, the labels are read from this file instead of the dataset
    crop_size           = None,
    sampler_kwargs = {},
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    if overwrite_cls_labels_path is not None:
        dataset_cls_labels = {}
        with open(overwrite_cls_labels_path, "r") as f:
            # import pdb; pdb.set_trace()
            # try:
            #     dataset_cls_labels = json.load(f)
            # except:
            #     # this should be a dictionary with keys the absolute filepaths and values the labels
            #     for line in f:
            #         json_item = json.loads(line)
            #         dataset_cls_labels[json_item['image_file']] = json_item['label']
            for line in f:
                json_item = json.loads(line)
                dataset_cls_labels[json_item['image_file']] = json_item['label']
    # import pdb; pdb.set_trace()

    # Load dataset.
    dist.print0('Loading dataset...')
    dataset_obj = ambient_utils.dataset.SyntheticallyCorruptedImageFolderDataset(**dataset_kwargs)
    # random indices for dataset visualization
    indices = [476716, 801177, 208667, 84697, 708005, 481119, 882784, 314948, 241315, 900832, 937237, 522057, 844026, 1021191, 789191, 668501]
    indices = [index % len(dataset_obj) for index in indices]
    images_to_save = [torch.tensor(dataset_obj[i]['image']) for i in indices]
    if dist.get_rank() == 0:
        ambient_utils.save_images(torch.stack(images_to_save), os.path.join(run_dir, "dataset.png"), save_wandb=True)
 
    # check if there is a file annotations.jsonl in the dataset_kwargs.path
    annotations_file = os.path.join(dataset_kwargs["path"], "annotations.jsonl")
    annotations = defaultdict(lambda: (0., 0.))
    lines_read = 0
    if os.path.exists(annotations_file):
        # read sigmas from the file
        sigmas_path = os.path.join(dataset_kwargs["path"], "sigmas.txt")
        # make sure that the the filepath exists
        if os.path.exists(sigmas_path):
            with open(sigmas_path, "r") as f:
                sigmas = [float(line.strip()) for line in f]
            device = "cuda" if torch.cuda.is_available() else "cpu"
            sigmas = torch.tensor(sigmas, device=device)
            sigmas = sigmas.sort(dim=0)[0]
        with open(annotations_file, "r") as f:
            for line in f:
                lines_read += 1
                line_json = json.loads(line)
                filename = line_json["filename"]
                # raw probabilities are stored that need to be processed
                if "probabilities" in line_json:
                    probs = line_json["probabilities"]
                    probs = np.array(probs).mean(axis=-1)
                    ema_probs = apply_ema(probs, window=cls_ema_window)
                    first_confusion = analyze_classifier_trajectory(torch.tensor(ema_probs).to(device), sigmas, epsilon=cls_epsilon)['first_confusion']
                    annotations[filename] = (first_confusion.cpu().item(), 0.)
                elif any(key.startswith("crop_predictions") for key in line_json):
                    patch_size_to_probs = {}
                    for key, value in line_json.items():
                        if key.startswith("crop_predictions"):
                            patch_size = int(key.split("_")[-1])
                            patch_size_to_probs[patch_size] = np.mean(value)
                    
                    # get the biggest crop size for which the probability is above 0.3
                    for patch_size in sorted(patch_size_to_probs.keys(), reverse=True):
                        if patch_size_to_probs[patch_size] > 0.25:
                            break
                    else:
                        patch_size = 1
                    
                    patch_to_sigma = {
                        1: 0.01,
                        4: 0.05,
                        8: 0.15,
                        16: 0.2,
                        24: 0.35,
                        32: 0.55,
                        48: 0.7,
                        64: 1.0,
                    }
                    sigma_max = patch_to_sigma[patch_size]
                    annotations[filename] = (300.0, sigma_max)

                # if single time
                elif "annotation" in line_json or "sigma" in line_json:
                    annotations[filename] = (line_json["annotation"], 0.) if "annotation" in line_json else (line_json["sigma"], 0)
                elif "sigma_min" in line_json and "sigma_max" in line_json:
                    annotations[filename] = (line_json["sigma_min"], line_json["sigma_max"])
                else:
                    raise ValueError(f"Could not parse line {line}")
    
    if dist.get_rank() == 0:
        # save the processed annotations to a file inside the run_dir
        with open(os.path.join(run_dir, "annotations_processed.jsonl"), "w") as f:
            for filename, annotation in annotations.items():
                f.write(f"{filename}: {annotation}\n")

        # print the number of annotations
        print(f"Num annotations: {len(list(annotations.keys()))}, Lines read: {lines_read}")
        # print the average min annotation
        print(f"Average min annotation: {np.mean([x[0] for x in annotations.values()])}")
        # print the average min annotation excluding values that are exactly 0
        print(f"Average min annotation excluding 0: {np.mean([x[0] for x in annotations.values() if (x[0] != 0 and x[0] != 300)])}")
        # print the average max annotation
        print(f"Average max annotation: {np.mean([x[1] for x in annotations.values()])}")
        # print the average max annotation excluding values that are exactly 0
        print(f"Average max annotation excluding 0: {np.mean([x[1] for x in annotations.values() if x[1] != 0])}")
    dataset_obj.annotations = dict(annotations)
    print('Rank', dist.get_rank(), 'Size', dist.get_world_size())
    print('Sampler kwargs', sampler_kwargs)
    dataset_sampler = misc.InfiniteSampler(
        dataset=dataset_obj, 
        rank=dist.get_rank(), 
        num_replicas=dist.get_world_size(), 
        seed=seed,
        **sampler_kwargs,
    )
    print('Constructed sampler')
    print(dataset_obj, dataset_sampler, batch_gpu, data_loader_kwargs)
    # data_loader_kwargs['num_workers'] = 0
    # data_loader_kwargs['prefetch_factor'] = None
    dataloader = torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs)
    print('Constructed dataloader')
    try:
        dataset_iterator = iter(dataloader)
    except RuntimeError as e:
        print("DataLoader failed:", e)
    print('Constructed dataloder iterator')
    
    # Construct network.
    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=dataset_obj.resolution, img_channels=dataset_obj.num_channels, label_dim=dataset_obj.label_dim)
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
    net.train().requires_grad_(True).to(device)
    with torch.no_grad():
        images = torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        sigma = torch.ones([batch_gpu], device=device)
        labels = torch.zeros([batch_gpu, net.label_dim], device=device)
        misc.print_module_summary(net, [images, sigma, labels], max_nesting=2, verbose=dist.get_rank() == 0)

    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs) # subclass of torch.optim.Optimizer
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None # training.augment.AugmentPipe
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], broadcast_buffers=True, find_unused_parameters=True)
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
        del data # conserve memory
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'), weights_only=False)
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
        optimizer.load_state_dict(data['optimizer_state'])
        del data # conserve memory

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    while True:

        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                dataset_item = next(dataset_iterator)
                images = dataset_item["image"].to(device)    
                labels = dataset_item["label"].to(device)  # attention: this is NOT the label for good or bad image. This is more like a class label (dog, cat, etc.)
                cls_labels = dataset_item.get("corruption_label", torch.zeros_like(labels)).to(device)

                if overwrite_cls_labels_path is not None:
                    cls_labels = torch.tensor([dataset_cls_labels[x] for x in dataset_item['filename']], device=device)
                
                batch_annotations = [annotations[x] for x in dataset_item['filename']]

                sigma_tn = torch.tensor([min(x[0], x[1]) for x in batch_annotations], device=device)
                sigma_t = torch.tensor([dataset_sampler.sampled_sigmas[x] for x in dataset_item['filename']], device=device)

                # ambient-crops: this means that the image is used in the (0, t_min) range
                sigma_tn = torch.where(sigma_tn > sigma_t, torch.zeros_like(sigma_tn), sigma_tn)

                x0, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
                x_tn = x0 + dataset_item["noise"].to(device).to(images.dtype) * sigma_tn[:, None, None, None]

                if crop_size is not None:
                    # get a random crop of the image
                    crop_start_h = torch.randint(0, images.shape[2] - crop_size + 1, (1,))
                    crop_end_h = crop_start_h + crop_size
                    crop_start_w = torch.randint(0, images.shape[3] - crop_size + 1, (1,))
                    crop_end_w = crop_start_w + crop_size
                    x_tn[:, :, :crop_start_h, :] = 0
                    x_tn[:, :, crop_end_h:, :] = 0
                    x_tn[:, :, :, :crop_start_w] = 0
                    x_tn[:, :, :, crop_end_w:] = 0
                    
                loss, x0_pred, sigma, noisy_input = loss_fn(net=ddp, x0=x0, x_tn=x_tn, sigma_tn=sigma_tn, sigma_t=sigma_t, labels=labels, augment_labels=augment_labels, cls_labels=cls_labels)
                bucket_indices = ambient_utils.utils.bucketize(sigma.squeeze(), 4)

                # every 100 steps save the images
                if cur_tick % 100 == 0 and dist.get_rank() == 0:
                    ambient_utils.save_images(images, os.path.join(run_dir, f"batch_images_{cur_tick}.png"), save_wandb=True)
                    ambient_utils.save_images(x_tn, os.path.join(run_dir, f"current_sigma_images_{cur_tick}.png"), save_wandb=True)
                    ambient_utils.save_images(noisy_input, os.path.join(run_dir, f"inputs_{cur_tick}.png"), save_wandb=True)
                    if 'CLS' not in network_kwargs.class_name:
                        ambient_utils.save_images(x0_pred, os.path.join(run_dir, f"outputs_{cur_tick}.png"), save_wandb=True)
                
                training_stats.report('Loss/loss', loss)

                for bucket_idx in range(4):
                    training_stats.report(f'Loss/bucket_{bucket_idx}', loss[bucket_indices == bucket_idx])

                loss.sum().mul(loss_scaling / batch_gpu_total).backward()

        torch.nn.utils.clip_grad_norm_(ddp.parameters(), clip)

        # Update weights.
        for g in optimizer.param_groups:
            g['lr'] = optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        optimizer.step()

        # Update EMA.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value # conserve memory
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                    pickle.dump(data, f)
            del data # conserve memory

        # Save full dump of the training state.
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
            torch.save(dict(net=net, optimizer_state=optimizer.state_dict()), os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'))

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            # report to wandb
            for key, value in training_stats.default_collector.as_dict().items():
                wandb.log({key: value}, step=cur_tick * snapshot_ticks)
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------

