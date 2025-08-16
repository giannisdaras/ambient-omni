#!/usr/bin/env python3
import os
import argparse
import torch
import open_clip
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
    parser.add_argument("--prompts", required=True,
                        help="CSV file with `image_id` and `text` columns")
    parser.add_argument("--output_csv", required=True,
                        help="Final merged CSV path")
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()

def preprocess_images(images, preprocess):
    # images: list of HxWxC numpy arrays
    return torch.cat([
        preprocess(T.ToPILImage()(torch.from_numpy(im))).unsqueeze(0)
        for im in images
    ], 0)

@torch.no_grad()
def compute_alignment(image, caption, model, preprocess, tokenizer, device):
    """
    Returns the cosine similarity between the image and caption.
    """
    # preprocess + batchify
    clip_input = preprocess_images([image], preprocess).to(device)
    # tokenize caption
    text_tokens = tokenizer([caption]).to(device)
    # encode & normalize
    img_feat = model.encode_image(clip_input, normalize=True)
    txt_feat = model.encode_text(text_tokens, normalize=True)
    # dot-product = cosine similarity
    return (img_feat * txt_feat).sum(dim=-1).cpu().item()

@torch.no_grad()
def compute_quality(image, model, preprocess, tokenizer, device):
    """
    Scores an image as “good” vs “bad” by softmax over two prompts.
    """
    clip_input = preprocess_images([image], preprocess).to(device)
    text_tokens = tokenizer(["good image", "bad image"]).to(device)
    # raw features (not normalized here—so scale applies)
    img_feat = model.encode_image(clip_input, normalize=True)
    txt_feat = model.encode_text(text_tokens, normalize=True)
    # compute logits = (image·text) * exp(logit_scale)
    logits = img_feat @ txt_feat.t() * model.logit_scale.exp()
    probs = logits.softmax(dim=-1)
    # probability of “good image”
    return probs[0, 0].cpu().item()

class SingleImageDataset(Dataset):
    def __init__(self, image_folder: ImageFolderDataset, prompts_df: pd.DataFrame):
        self.folder = image_folder
        self.prompts = prompts_df

    def __len__(self):
        return len(self.folder)

    def __getitem__(self, idx):
        sample = self.folder[idx]
        filename = sample['filename']
        image_id = int(filename.replace('.png', ''))
        caption = self.prompts.loc[
            self.prompts.image_id == image_id, "text"
        ].item()
        return filename, sample['image'], image_id, caption

def main():
    args = parse_args()
    torch.distributed.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # load OpenCLIP model + tokenizer from HF Hub
    model, preprocess = open_clip.create_model_from_pretrained(
        "hf-hub:apple/DFN5B-CLIP-ViT-H-14-378"
    )  # :contentReference[oaicite:0]{index=0}
    tokenizer = open_clip.get_tokenizer("hf-hub:apple/DFN5B-CLIP-ViT-H-14-378")
    model.to(device).eval()

    # build dataset + loader
    prompts_df = pd.read_csv(args.prompts)
    dataset = SingleImageDataset(ImageFolderDataset(args.images), prompts_df)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    # per-rank checkpoint file
    base, ext = os.path.splitext(args.output_csv)
    rank_csv = f"{base}_{rank}{ext}"
    done_ids = set()
    if os.path.exists(rank_csv):
        done_ids = set(pd.read_csv(rank_csv).image_id.tolist())

    # iterate
    for filename, image, image_id, caption in tqdm(loader, disable=(rank != 0)):
        image_id = image_id.item()
        if image_id in done_ids:
            continue

        img_arr = image[0].numpy()
        align_score = compute_alignment(
            img_arr, caption[0], model, preprocess, tokenizer, device
        )
        quality_score = compute_quality(
            img_arr, model, preprocess, tokenizer, device
        )

        row = {
            "image_path": filename[0],
            "image_id": image_id,
            "caption": caption[0],
            "clip_alignment": align_score,
            "clip_quality": quality_score,
        }
        pd.DataFrame([row]).to_csv(
            rank_csv, mode="a",
            header=not os.path.exists(rank_csv),
            index=False
        )

    # synchronize and merge
    torch.distributed.barrier()
    if rank == 0:
        dfs = []
        for r in range(world_size):
            p = f"{base}_{r}{ext}"
            if os.path.exists(p):
                dfs.append(pd.read_csv(p))
        merged = pd.concat(dfs, ignore_index=True)
        merged.to_csv(args.output_csv, index=False)
        print(f"[rank 0] Merged results → {args.output_csv}")

if __name__ == "__main__":
    main()
