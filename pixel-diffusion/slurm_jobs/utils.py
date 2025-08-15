import os
import glob
import json
import sys
import inspect
from pathlib import Path
import pickle
import math
import shutil
import torch
from typing import List

try:
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)
except ImportError as e:
    print(e)

def setup():
    setup_cmd = """source ~/.bashrc
cd /home1/07362/gdaras/ambient/pixel-diffusion
conda activate ambient-syn
source .env
module use /home1/00422/cazes/modulefiles
module load ibrun
"""
    print(setup_cmd)
    return setup_cmd


def cleanup_empty_folders():
    base_dir = os.environ["BASE_PATH"]
    num_empty_folders = 0
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            pkl_files = glob.glob(os.path.join(folder_path, "*.pkl"))
            pt_files = glob.glob(os.path.join(folder_path, "*.pt"))
            training_options = glob.glob(os.path.join(folder_path, "training_options.json"))
            if not (pkl_files or pt_files) and training_options:
                num_empty_folders += 1
                shutil.rmtree(folder_path)
                num_empty_folders += 1
    print(f"Removed {num_empty_folders} runs that were empty.")




def find_training_folders_based_on_params(
                                         noise_config: str = None,
                                         corruption_probability: float = 0.5,
                                         dataset_keep_percentage: float = 1.0, 
                                         dataset: str="imagenet",  # dataset name
                                         weight_decay=0.0,
                                         dataset_path: str = None,  # This is useful if we want to make sure we have the correct annotated version of the dataset.
                                         debug=False,
                                         glob_only_dataset=False,
                                         ):
    base_dir = os.environ["BASE_PATH"]
    found_folders = []
    if debug:
        print("Searching for folders with the given params...")
        print(f"Noise config: {noise_config}")
        print(f"Corruption probability: {corruption_probability}")
        print(f"Dataset keep percentage: {dataset_keep_percentage}")
        print(f"Dataset: {dataset}")
        print(f"Weight decay: {weight_decay}")
        print(f"Dataset path: {dataset_path}")
    dirs_to_search = glob.glob(os.path.join(base_dir, f"*{dataset}*")) if glob_only_dataset else glob.glob(os.path.join(base_dir, "*"))
    for folder in dirs_to_search:
        # each folder has a training_options.json file that we need to open
        # and check if the sigma and corruption_probability match
        try:
            with open(os.path.join(folder, "training_options.json"), encoding="utf-8") as f:
                options = json.load(f)
                found_noise_config = options["dataset_kwargs"].get("noise_config", None)
                # print(f"Found noise config: {found_noise_config} of dataset {dataset}")
                found_corruption_probability = options["dataset_kwargs"].get("corruption_probability", None)
                dataset_options = options["dataset_kwargs"]
                found_dataset_path = dataset_options["path"]
                optimizer_options = options["optimizer_kwargs"]
                found_dataset_keep_percentage = float(dataset_options["dataset_keep_percentage"]) if "dataset_keep_percentage" in dataset_options else 1.0
                found_weight_decay = float(optimizer_options["weight_decay"]) if "weight_decay" in optimizer_options else 0.0
                if (
                    found_dataset_keep_percentage == dataset_keep_percentage and
                    found_weight_decay == weight_decay and
                    (found_noise_config == noise_config or corruption_probability == 0) and
                    found_corruption_probability == corruption_probability and
                    (os.path.normpath(found_dataset_path) == os.path.normpath(dataset_path)) or (dataset_path is None)
                ):
                        found_folders.append(folder)
                        continue
        except Exception as e:
            print(f"Error loading training options for {folder}: {e}")
            continue
    if debug:
        print(f"Found {len(found_folders)} folders with the given params.")
        print(found_folders)
    return found_folders

def find_latest_checkpoint(folders: List[str], pkl=False):
    checkpoints = [file for folder in folders for file in glob.glob(os.path.join(folder, "*.pkl" if pkl else "*.pt"))]
    if checkpoints:
        checkpoint_path = max(checkpoints, key=os.path.getctime)
        return checkpoint_path

def find_nearest_checkpoint(folders: List[str], checkpoint_index: int, pkl=False):
    checkpoints = [file for folder in folders for file in glob.glob(os.path.join(folder, "*.pkl" if pkl else "*.pt"))]
    if checkpoints:
        nearest_checkpoint = min(checkpoints, key=lambda x: abs(int(x.split("-")[-1].split(".")[0]) - checkpoint_index))
        return nearest_checkpoint



if __name__ == "__main__":
    print(find_training_folders_based_on_params(noise_config="blurs", 
                                                corruption_probability=0.0, 
                                                dataset_keep_percentage=1.0, 
                                                dataset="cifar",
                                                dataset_path="/scratch/07362/gdaras/datasets/cifar-fixed-sigma-0.2-0.0-blurs0_4-0.9/",
                                                # dataset_path="/scratch/07362/gdaras/datasets/cifar10-32x32.zip",
                                                # dataset_path="/scratch/07362/gdaras/datasets/cifar-freq-single-time-blurs0_8-0.9/"
                                                # dataset_path="/scratch/07362/gdaras/datasets/cifar-freq-single-time-blurs1-0.9/"
                                                debug=False
                                                )
        )
