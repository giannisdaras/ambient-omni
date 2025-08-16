import os
import torch
import torchvision
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from ambient_utils import dist_utils
from micro_diffusion.models.model import create_latent_diffusion
import numpy as np
import random
from streaming import StreamingDataset, Stream
from huggingface_hub import hf_hub_download
from safetensors import safe_open

def linear_vendi_diversity(images: torch.Tensor, device: torch.device = None) -> float:
    device = device or images.device
    B = images.size(0)
    images = images.to(device)

    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    model.head = torch.nn.Identity()
    model.eval().to(device)

    imgs = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
    imgs = (imgs - mean) / std

    with torch.no_grad():
        feats = model(imgs)
        feats = F.normalize(feats, dim=1)

    K = feats @ feats.T
    K_norm = K / B
    eigs = torch.linalg.eigvalsh(K_norm)
    eigs = torch.clamp(eigs, min=0)
    probs = eigs
    H = - (probs * (probs + 1e-12).log()).sum()
    return H.exp().item()

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--guidance_scale', type=float, default=1.5)
parser.add_argument('--num_inference_steps', type=int, default=30)
parser.add_argument('--dataset_path', type=str, required=True)
parser.add_argument('--output_path', type=str, default=None)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--num_prompts', type=int, default=100)  # MODIFIED
parser.add_argument('--num_images_per_prompt', type=int, default=16)  # MODIFIED

def main(args):
    rank = dist_utils.get_rank()
    world_size = dist_utils.get_world_size()
    torch.manual_seed(args.seed + rank)

    if args.output_path is None:
        model_dir = os.path.dirname(args.model_path)
        model_tag = os.path.basename(args.model_path).replace('.pt', '') + f'_gs{args.guidance_scale}'
        args.output_path = os.path.join(model_dir, 'generations', model_tag)

    if rank == 0:
        os.makedirs(args.output_path, exist_ok=True)
        print("Generating images in:", args.output_path)

    torch.distributed.barrier()

    streams = [Stream(remote=None, local=args.dataset_path)]
    dataset = StreamingDataset(streams=streams, shuffle=False)

    rng = np.random.RandomState(args.seed) 
    sampled_indices = rng.choice(30_000, size=args.num_prompts, replace=False)
    selected_captions = [dataset[i]['caption'] for i in sampled_indices]
    selected = pd.DataFrame({
        'image_id': sampled_indices,
        'text': selected_captions
    })

    # Resume state
    rank_csv_path = os.path.join(args.output_path, f"vendi_rank{rank}.csv")
    completed_indices = set()
    if os.path.exists(rank_csv_path):
        try:
            completed_df = pd.read_csv(rank_csv_path)
            completed_indices = set(completed_df["prompt_index"].tolist())
        except Exception as e:
            print(f"[Rank {rank}] Failed to read existing CSV: {e}")

    model = create_latent_diffusion(latent_res=64, in_channels=4, pos_interp_scale=2.0).to('cuda')
    if args.model_path.startswith('giannisdaras/'):
        model_dict_path = hf_hub_download(repo_id=args.model_path, filename="model.safetensors")

        model_dict = {}
        with safe_open(model_dict_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                model_dict[key] = f.get_tensor(key)
        
        float_model_params = {
            k: v.to(torch.float32) for k, v in model_dict.items()
        }
        model.dit.load_state_dict(float_model_params)
    else:
        checkpoint = torch.load(args.model_path, map_location='cuda', weights_only=False)
        if 'state' in checkpoint:
            model_dict = checkpoint['state']['model']
            # Convert parameters to float32
            float_model_params = {
                k.replace('dit.', ''): v.to(torch.float32) for k, v in model_dict.items() if 'dit' in k
            }
            model.dit.load_state_dict(float_model_params)
        else:
            model_dict = checkpoint
            model.dit.load_state_dict(model_dict)
    print("Model loaded!")

    for idx in range(rank, len(selected), world_size):
        image_id = selected.image_id[idx]
        if image_id in completed_indices:
            print(f"[Rank {rank}] Skipping prompt index {idx} (already processed)")
            continue

        prompt = selected.text[idx]
        subfolder = os.path.join(args.output_path, str(image_id))
        os.makedirs(subfolder, exist_ok=True)

        with torch.no_grad():
            images = model.generate(
                prompt=[prompt] * args.num_images_per_prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed + idx
            )
        for i, image in enumerate(images):
            torchvision.utils.save_image(image, os.path.join(subfolder, f"{i}.png"))

        vendi_score = linear_vendi_diversity(images, device=torch.device("cuda"))
        print(f"[Rank {rank}] [{image_id}] \"{prompt}\" | Vendi diversity: {vendi_score:.4f}")

        with open(rank_csv_path, 'a') as f:
            if os.stat(rank_csv_path).st_size == 0:
                f.write("image_id,caption,vendi_diversity\n")
            f.write(f"{image_id},\"{prompt}\",{vendi_score:.6f}\n")

    # Merge and cleanup
    torch.distributed.barrier()
    if rank == 0:
        merged_path = os.path.join(args.output_path, "vendi_diversities.csv")
        all_dfs = []
        for r in range(world_size):
            per_rank_path = os.path.join(args.output_path, f"vendi_rank{r}.csv")
            if os.path.exists(per_rank_path):
                try:
                    df = pd.read_csv(per_rank_path)
                    all_dfs.append(df)
                    os.remove(per_rank_path)  # Cleanup
                except Exception as e:
                    print(f"Error loading or deleting {per_rank_path}: {e}")
        merged = pd.concat(all_dfs, ignore_index=True)
        merged.to_csv(merged_path, index=False)
        print(f"Merged Vendi CSV written to: {merged_path}")
        print(f"Average Vendi Diversity: {merged['vendi_diversity'].mean():.4f}")



if __name__ == "__main__":
    args = parser.parse_args()
    torch.multiprocessing.set_start_method('spawn')
    dist_utils.init()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    if torch.distributed.is_initialized():
        torch.distributed.barrier(device_ids=[local_rank])
    dist_utils.print0("Total nodes:", dist_utils.get_world_size())
    main(args)
