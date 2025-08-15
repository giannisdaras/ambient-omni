#!/usr/bin/env python3
import os
import argparse
import torch
import clip
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from ambient_utils import ImageFolderDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True,
                        help="Path to the image directory")
    parser.add_argument("--num_images", type=int, default=5000,
                        help="Number of images to evaluate")
    parser.add_argument("--output_csv", required=True,
                        help="Final merged CSV path")
    return parser.parse_args()

def preprocess_images(images, preprocess):
    # images: list of HxWxC numpy arrays
    return torch.cat([
        preprocess(T.ToPILImage()(torch.from_numpy(im))).unsqueeze(0)
        for im in images
    ], 0)

import piq
clip_iqa = piq.CLIPIQA(data_range=1.0).cuda()

@torch.no_grad()
def compute_quality(image, model, preprocess, device):
    return clip_iqa(torch.tensor(image).unsqueeze(0).cuda()).item()

class SingleImageDataset(Dataset):
    def __init__(self, image_folder: ImageFolderDataset):
        self.folder = image_folder

    def __len__(self):
        return len(self.folder)

    def __getitem__(self, idx):
        sample = self.folder[idx]
        filename = sample['filename']
        image_id = int(filename.replace('.png', ''))
        return filename, sample['image'], image_id

def main():
    args = parse_args()
    torch.distributed.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # load CLIP once
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # build dataset + loader
    dataset = SingleImageDataset(ImageFolderDataset(args.images))
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler)

    # iterate
    index = 0
    for filename, image, image_id in tqdm(loader, disable=(rank != 0), total=args.num_images):
        image_id = image_id.item()
        # if image_id in done_ids:
        #     continue
        if index >= args.num_images:
            break
        index += 1

        img_arr = image[0].numpy()
        quality_score = compute_quality(img_arr, model, preprocess, device, args.labels)

        row = {
            "image_path": filename[0],
            "image_id": image_id,
            # "caption": caption[0],
            # "clip_alignment": align_score,
            "clip_quality": quality_score,
        }
        pd.DataFrame([row]).to_csv(
            args.output_csv, mode="a",
            header=not os.path.exists(args.output_csv),
            index=False
        )




if __name__ == "__main__":
    main()