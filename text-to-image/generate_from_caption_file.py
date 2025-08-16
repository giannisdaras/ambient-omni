import os
from micro_diffusion.models.model import create_latent_diffusion
from micro_diffusion.datasets.latents_loader import build_streaming_latents_dataloader
import ambient_utils
# from reconstruct_phema import pt_path
import torch
import torchvision
import argparse
from typing import List, Optional
from functools import partial
from tqdm import tqdm
import pandas as pd
from ambient_utils import dist_utils
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from micro_diffusion.models.utils import (
    DATA_TYPES,
    DistLoss,
    UniversalTextEncoder,
    UniversalTokenizer,
    text_encoder_embedding_format,
)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--guidance_scale', type=float, default=1.5, help='Guidance scale for classifier-free guidance')
parser.add_argument('--num_inference_steps', type=int, default=30, help='Number of inference steps')
parser.add_argument('--dataset_captions_path', type=str, default="/scratch/07362/gdaras/MS-COCO_val2014_30k_captions.csv")
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--seed', type=int, default=42, help='Seed')
parser.add_argument('--output_path', type=str, default=None)

def main(args):
    if args.output_path is None:
        model_path = '/'.join(args.model_path.split('/')[:-1])
        model_name = args.model_path.split('/')[-1].replace('.pt', '') + 'guidance' + str(args.guidance_scale)
        output_path = os.path.join(model_path, 'generations', model_name)
    else:
        output_path = args.output_path

    
    # Distributed
    rank = dist_utils.get_rank()
    world_size = dist_utils.get_world_size()

    if rank == 0:
        if args.output_path is None:
            os.makedirs(os.path.join(model_path, 'generations'), exist_ok=True)
            os.makedirs(os.path.join(model_path, 'generations', model_name), exist_ok=True)
            print("generating images in: ", os.path.join(model_path, 'generations', model_name))
        else:
            print("generating images in: ", output_path)

    # Loader
    print("Loading ids and captions")
    dataset_ids_and_captions = pd.read_csv(args.dataset_captions_path)
    print("Ids and captions loaded")

    # Model creation
    params = {
        'latent_res': 64,
        'in_channels': 4,
        'pos_interp_scale': 2.0,
    }
    model = create_latent_diffusion(**params).to('cuda')
    print("Model creation done!")

    # Model loading
    print("Loading model...")
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

    # Make folder
    if rank == 0:
        os.makedirs(output_path, exist_ok=True)
    torch.distributed.barrier()

    # Generation
    process_batches = np.array(list(range(0, len(dataset_ids_and_captions), args.batch_size)))[rank::world_size]
    for i in tqdm(process_batches):
        image_ids = list(dataset_ids_and_captions.image_id[i:i+args.batch_size])
        prompts = list(dataset_ids_and_captions.text[i:i+args.batch_size])
        if not os.path.exists(os.path.join(output_path, str(image_ids[-1])+'.png')):
            with torch.no_grad():
                # seed=args.seed
                images = model.generate(prompt=prompts, num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale, seed=int(i))
            for id, image in zip(image_ids, images):
                torchvision.utils.save_image(image, os.path.join(output_path, str(id)+'.png'))
        else:
            print(f'Rank {rank} skipping batch {i}')

if __name__ == "__main__":
    args = parser.parse_args()
    # Distributed
    torch.multiprocessing.set_start_method('spawn')
    dist_utils.init()
    # Explicitly set device ID for each process to avoid ProcessGroupNCCL warnings
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    # Force PyTorch to use this device for barrier operations
    if torch.distributed.is_initialized():
        torch.distributed.barrier(device_ids=[local_rank])
    dist_utils.print0("Total nodes: ", dist_utils.get_world_size())
    main(args)