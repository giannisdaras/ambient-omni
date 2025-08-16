from functools import partial
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from composer.models import ComposerModel
from diffusers import AutoencoderKL
from easydict import EasyDict

from . import dit as model_zoo
from .utils import (
    DATA_TYPES,
    DistLoss,
    UniversalTextEncoder,
    UniversalTokenizer,
    text_encoder_embedding_format,
)

import ambient_utils
from micro_diffusion.models.model import LatentDiffusion


class AmbientDiffusion(LatentDiffusion):
    """Latent diffusion model that generates images from text prompts.

    This model combines a DiT (Diffusion Transformer) model for denoising image latents,
    a VAE for encoding/decoding images to/from the latent space, and a text encoder
    for converting text prompts into embeddings. It implements the EDM (Elucidated
    Diffusion Model) sampling process.

    Args:
        dit (nn.Module): Diffusion Transformer model
        vae (AutoencoderKL): VAE model from diffusers for encoding/decoding images
        text_encoder (UniversalTextEncoder): Text encoder for converting prompts to embeddings
        tokenizer (UniversalTokenizer): Tokenizer for processing text prompts
        image_key (str, optional): Key for image data in batch dict. Defaults to 'image'.
        text_key (str, optional): Key for text data in batch dict. Defaults to 'captions'.
        image_latents_key (str, optional): Key for precomputed image latents in batch dict. Defaults to 'image_latents'.
        text_latents_key (str, optional): Key for precomputed text latents in batch dict. Defaults to 'caption_latents'.
        precomputed_latents (bool, optional): Whether to use precomputed latents (must be in the batch). Defaults to True.
        dtype (str, optional): Data type for model ops. Defaults to 'bfloat16'.
        latent_res (int, optional): Resolution of latent space assuming 8x downsampling by VAE. Defaults to 32.
        p_mean (float, optional): EDM log-normal noise mean. Defaults to -0.6.
        p_std (float, optional): EDM log-normal noise standard-deviation. Defaults to 1.2.
        train_mask_ratio (float, optional): Ratio of patches to mask during training. Defaults to 0.
    """

    def __init__(
        self,
        dit: nn.Module,
        vae: AutoencoderKL,
        text_encoder: UniversalTextEncoder,
        tokenizer: UniversalTokenizer,
        image_key: str = 'image',
        text_key: str = 'captions',
        image_latents_key: str = 'image_latents',
        text_latents_key: str = 'caption_latents',
        precomputed_latents: bool = True,
        dtype: str = 'bfloat16',
        latent_res: int = 32,
        p_mean: float = -0.6,
        p_std: float = 1.2,
        train_mask_ratio: float = 0.
    ):
        super().__init__(
            dit=dit,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            image_key=image_key,
            text_key=text_key,
            image_latents_key=image_latents_key,
            text_latents_key=text_latents_key,
            precomputed_latents=precomputed_latents,
            dtype=dtype,
            latent_res=latent_res,
            p_mean=p_mean,
            p_std=p_std,
            train_mask_ratio=train_mask_ratio,
        )

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get image latents
        if self.precomputed_latents and self.image_latents_key in batch:
            # Assuming that latents have already been scaled, i.e., multiplied with the scaling factor
            latents = batch[self.image_latents_key]
        else:
            with torch.no_grad():
                images = batch[self.image_key]
                latents = self.vae.encode(
                    images.to(DATA_TYPES[self.dtype])
                )['latent_dist'].sample().data
                latents *= self.latent_scale

        # Get text embeddings
        if self.precomputed_latents and self.text_latents_key in batch:
            conditioning = batch[self.text_latents_key]
        else:
            captions = batch[self.text_key]
            captions = captions.view(-1, captions.shape[-1])
            if 'attention_mask' in batch:
                conditioning = self.text_encoder.encode(
                    captions,
                    attention_mask=batch['attention_mask'].view(-1, captions.shape[-1])
                )[0]
            else:
                conditioning = self.text_encoder.encode(captions)[0]

        # Zero out dropped captions. Needed for classifier-free guidance during inference.
        if 'drop_caption_mask' in batch.keys():
            conditioning *= batch['drop_caption_mask'].view(
                [-1] + [1] * (len(conditioning.shape) - 1)
            )

        # Get ambient information
        sigma_tn = batch.get('sigma_tn', torch.zeros(latents.shape[0], device=latents.device).unsqueeze(1))
        noise = batch.get('noise', torch.zeros_like(latents))
        sigma_t = batch.get('sigma_t', (torch.randn(latents.shape[0], device=latents.device) * self.edm_config.P_std + self.edm_config.P_mean).unsqueeze(1).exp())      

        loss = self.ambient_loss(
            latents.float(),
            conditioning.float(),
            sigma_t=sigma_t,
            sigma_tn=sigma_tn,
            noise=noise,
            mask_ratio=self.train_mask_ratio if self.training else self.eval_mask_ratio
        )
        return (loss, latents, conditioning)

    def ambient_loss(self, x: torch.Tensor, y: torch.Tensor, sigma_t: torch.Tensor, sigma_tn: torch.Tensor, noise: torch.Tensor, mask_ratio: float = 0, **kwargs) -> torch.Tensor:
        # Weight
        weight = (
            (sigma_t ** 2 + self.edm_config.sigma_data ** 2) /
            (sigma_t * self.edm_config.sigma_data) ** 2
        )
        ambient_factor = (sigma_t ** 4) / ((sigma_t ** 2 - sigma_tn ** 2)**2)
        edm_weight = (self.edm_config.sigma_data ** 2 + sigma_t ** 2) / ((self.edm_config.sigma_data ** 2) * (sigma_t ** 2))
        ambient_weight = ambient_factor * edm_weight

        # Forward ambient diffusion
        x_tn = x + sigma_tn[:, None, None] * noise
        x_t = x_tn + torch.sqrt(sigma_t**2 - sigma_tn**2)[:, None, None] * self.randn_like(x)

        # Model prediction
        model_out = self.model_forward_wrapper(x_t, sigma_t, y, self.dit, mask_ratio=mask_ratio,**kwargs)
        D_xn = model_out['sample']
        
        # Ambient loss
        D_xn_tn = ambient_utils.from_x0_pred_to_xnature_pred_ve_to_ve(D_xn, x_t, sigma_t, sigma_tn)
        loss = ambient_weight[:, None, None] * ((D_xn_tn - x_tn) ** 2)  # (N, C, H, W)

        if mask_ratio > 0:
            # Masking is not feasible during image generation as it only returns denoised version
            # for non-masked patches. Image generation requires all patches to be denoised.
            assert (
                self.dit.training and 'mask' in model_out
            ), 'Masking is only recommended during training'
            loss = F.avg_pool2d(loss.mean(dim=1), self.dit.patch_size).flatten(1)
            unmask = 1 - model_out['mask']
            loss = (loss * unmask).sum(dim=1) / unmask.sum(dim=1)  # (N,)
        return loss.mean()
    
    
def create_ambient_diffusion(
    vae_name: str = 'stabilityai/stable-diffusion-xl-base-1.0',
    text_encoder_name: str = 'openclip:hf-hub:apple/DFN5B-CLIP-ViT-H-14-378', 
    dit_arch: str = 'MicroDiT_XL_2',
    latent_res: int = 32,
    in_channels: int = 4,
    pos_interp_scale: float = 1.0,
    dtype: str = 'bfloat16',
    precomputed_latents: bool = True,
    p_mean: float = -0.6,
    p_std: float = 1.2,
    train_mask_ratio: float = 0.
) -> LatentDiffusion:
    # retrieve max sequence length (s) and token embedding dim (d) from text encoder
    s, d = text_encoder_embedding_format(text_encoder_name)

    dit = getattr(model_zoo, dit_arch)(
        input_size=latent_res,
        caption_channels=d,
        pos_interp_scale=pos_interp_scale,
        in_channels=in_channels
    )

    vae = AutoencoderKL.from_pretrained(
        vae_name,
        subfolder=None if vae_name=='ostris/vae-kl-f8-d16' else 'vae',
        torch_dtype=DATA_TYPES[dtype],
        pretrained=True
    )

    text_encoder = UniversalTextEncoder(
        text_encoder_name,
        dtype=dtype,
        pretrained=True
    )
    tokenizer = UniversalTokenizer(text_encoder_name)

    model = AmbientDiffusion(
        dit=dit,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        precomputed_latents=precomputed_latents,
        dtype=dtype,
        latent_res=latent_res,
        p_mean=p_mean,
        p_std=p_std,
        train_mask_ratio=train_mask_ratio
    )
    return model