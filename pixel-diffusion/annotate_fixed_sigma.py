"""
python annotate_freq.py --annotated_dataset_path=tmp_annotate --dataset_path=$CIFAR_PATH --corruption_probability=0.9
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
parser = argparse.ArgumentParser()
parser.add_argument("--annotated_dataset_path", type=str, required=True, help="Path to save the annotated dataset.")
parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset to annotate.")
parser.add_argument("--inference_noise_config", type=str, default="noise3", help="Corruption configuration.")
parser.add_argument("--corruption_probability", type=float, default=0.5, 
                    help="Corruption probability. This should be set to 0.0 if the dataset to be annotated is already corrupted.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
parser.add_argument("--min_fixed_sigma", type=float, default=3.0, help="The images can be used it they have at minimum this amount of noise.")
parser.add_argument("--max_fixed_sigma", type=float, default=0.0, help="The images can be used it they have at most this amount of noise.")
parser.add_argument("--save_only_clean", action="store_true", help="Save only clean images.")

def main(args):
    torch.multiprocessing.set_start_method('spawn')
    dist_utils.init()
    
    dist_utils.print0("Total nodes: ", dist_utils.get_world_size())
    if dist_utils.get_rank() == 0:
        os.makedirs(args.annotated_dataset_path, exist_ok=True)

    
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

        image_name = dataset_item['filename'][0].split("/")[-1]
        image_path = os.path.join(args.annotated_dataset_path, image_name)
        save_image(images[0], image_path)
        # print("Saved image: ", image_path)


        # Write to process-specific annotation file
        process_id = torch.distributed.get_rank()
        process_annotations_path = os.path.join(args.annotated_dataset_path, f"annotations_{process_id}.jsonl")
        with open(process_annotations_path, "a", encoding="utf-8") as f:
            annotation = {
                "filename": image_name,
                "sigma_min": args.min_fixed_sigma if dataset_item['corruption_label'].sum() > 0 else 0.0,
                "sigma_max": args.max_fixed_sigma if dataset_item['corruption_label'].sum() > 0 else 0.0,
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
