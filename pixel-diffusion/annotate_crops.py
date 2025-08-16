"""
python annotate_crops.py \
    --annotated_dataset_path=./outputs/annotated_afhq/afhq-dogs-help/ \
    --dataset_path=./data/afhqv2-64x64-partitioned/1/ \
    --corruption_probability=0.9 \
    --flip_probs \
    --checkpoint_paths=./outputs/ambient-syn-runs/00039-afhqv2-64x64-partitioned-uncond-ddpmpp-edmcls-gpus8-batch512-fp32-SIPct/network-snapshot-020070.pkl,./outputs/ambient-syn-runs/00037-afhqv2-64x64-partitioned-uncond-ddpmpp-edmcls-gpus8-batch512-fp32-uoJee/network-snapshot-020070.pkl,./outputs/ambient-syn-runs/00040-afhqv2-64x64-partitioned-uncond-ddpmpp-edmcls-gpus8-batch512-fp32-DtoBD/network-snapshot-020070.pkl
"""
import argparse
import numpy as np
from ambient_utils import save_image
from ambient_utils import dist_utils
from ambient_utils.dataset import SyntheticallyCorruptedImageFolderDataset
from ambient_utils.classifier import get_classifier_trajectory
import torch
from torch_utils.misc import copy_params_and_buffers
import pickle
import dnnlib
import os
import json
import importlib
import math
import portalocker
from tqdm import tqdm
import torchvision
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_paths", type=str, required=True, help="List of checkpoint paths (comma separated)")
parser.add_argument("--annotated_dataset_path", type=str, required=True, help="Path to save the annotated dataset.")
parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset to annotate.")
parser.add_argument("--inference_noise_config", type=str, default="identity", help="Corruption configuration.")
parser.add_argument("--corruption_probability", type=float, default=0.5, 
                    help="Corruption probability. This should be set to 0.0 if the dataset to be annotated is already corrupted.")
parser.add_argument("--flip_probs", default=False, help="Flip the probabilities.", action="store_true")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
parser.add_argument("--save_only_clean", action="store_true", help="Save only clean images.")

def create_patches(image, crop_size):
    patches = []
    _, h, w = image.shape
    for i in range(0, h, crop_size):
        for j in range(0, w, crop_size):
            patch = torch.zeros_like(image)
            patch[:, i:i+crop_size, j:j+crop_size] = image[:, i:i+crop_size, j:j+crop_size]
            patches.append(patch)
    return patches

def load_net_from_pkl(ckpt_file):
    base_folder = os.path.dirname(ckpt_file)
    with open(os.path.join(base_folder, "training_options.json"), "r", encoding="utf-8") as f:
        options = json.load(f)

    interface_kwargs = dict(img_resolution=options['dataset_kwargs']['resolution'], img_channels=3, label_dim=0)
    
    net = dnnlib.util.construct_class_by_name(**options['network_kwargs'], **interface_kwargs)
    with dnnlib.util.open_url(ckpt_file) as f:
        data = pickle.load(f)
    copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
    return net, options

def main(args):
    torch.multiprocessing.set_start_method('spawn')
    dist_utils.init()
    
    dist_utils.print0("Total nodes: ", dist_utils.get_world_size())
    if dist_utils.get_rank() == 0:
        os.makedirs(args.annotated_dataset_path, exist_ok=True)
        print("Will save annotations to: ", args.annotated_dataset_path)

    crop_sizes = []
    nets = []
    for checkpoint_path in args.checkpoint_paths.split(","):
        net, training_options = load_net_from_pkl(checkpoint_path)
        net.eval().to("cuda")
        nets.append(net)

        # Extract the crop_size from the training options
        crop_size = training_options.get('crop_size', None)
        crop_sizes.append(crop_size)
    print('crop sizes', crop_sizes)

    # prepare params for synthetic dataset corruption
    corruptions_dict = importlib.import_module(f"noise_configs.inference.{args.inference_noise_config}").corruptions_dict
    options = {}
    options['dataset_kwargs'] = {
        "path": args.dataset_path,
    }
    options['dataset_kwargs']['corruptions_dict'] = corruptions_dict
    options['dataset_kwargs']['corruption_probability'] = args.corruption_probability

    dataset_obj = SyntheticallyCorruptedImageFolderDataset(**options['dataset_kwargs'])
    dataset_loader = torch.utils.data.DataLoader(
            dataset_obj,
            batch_size=1,
            shuffle=False,
            sampler=torch.utils.data.distributed.DistributedSampler(dataset_obj, shuffle=False)
            )

    # all models wait for the path to be created
    torch.distributed.barrier()
    for dataset_item in tqdm(dataset_loader):
        
        if args.save_only_clean and dataset_item['corruption_label'].sum():
            # print("Skipping image: ", dataset_item['filename'][0])
            continue

        images = dataset_item["image"] * 2 - 1
        images = torchvision.transforms.Resize((64, 64))(images)

        image_name = dataset_item['filename'][0].split("/")[-1]
        image_path = os.path.join(args.annotated_dataset_path, image_name)
        save_image(images[0], image_path)

        crop_to_net_predictions = {}
        for net, crop_size in zip(nets, crop_sizes):
            padded_patches = create_patches(images[0], crop_size)

            # Get network predictions for all patches in batches
            patches_tensor = torch.stack(padded_patches).to("cuda")
            with torch.no_grad():
                logits = net(patches_tensor, 0.001 * torch.ones(len(patches_tensor), device="cuda"))["cls_logits"]
            crop_to_net_predictions["crop_predictions_" + str(crop_size)] = torch.sigmoid(logits).tolist()
            if args.flip_probs:
                crop_to_net_predictions["crop_predictions_" + str(crop_size)] = [1 - p for p in crop_to_net_predictions["crop_predictions_" + str(crop_size)]]
            

        # Write to process-specific annotation file
        process_id = torch.distributed.get_rank()
        process_annotations_path = os.path.join(args.annotated_dataset_path, f"annotations_{process_id}.jsonl")
        with open(process_annotations_path, "a", encoding="utf-8") as f:
            annotation = {
                "filename": image_name,
                **crop_to_net_predictions,
            }
            f.write(json.dumps(annotation) + "\n")

    # After all processes finish, merge files
    process_id = torch.distributed.get_rank()
    torch.distributed.barrier()
    if process_id == 0:
        # Merge all process files into single annotations file
        final_annotations_path = os.path.join(args.annotated_dataset_path, "annotations.jsonl")
        with open(final_annotations_path, "w", encoding="utf-8") as outfile:
            world_size = torch.distributed.get_world_size()
            for pid in range(world_size):
                proc_file = os.path.join(args.annotated_dataset_path, f"annotations_{pid}.jsonl")
                with open(proc_file) as infile:
                    outfile.write(infile.read())
                os.remove(proc_file)  # Clean up process file
        






if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
