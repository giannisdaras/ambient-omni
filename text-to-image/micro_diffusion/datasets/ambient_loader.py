import torch
import numpy as np
from streaming import Stream, StreamingDataset
from torch.utils.data import DataLoader
from typing import List, Dict, Union, Optional
from easydict import EasyDict

# Iterator
import os
from streaming.base.dataset import _Iterator
from concurrent.futures import ThreadPoolExecutor, wait, Future
from threading import Event
from time import sleep
from typing import Any, Iterator, Optional, Sequence, Union
import numpy as np
from filelock import FileLock
from numpy.typing import NDArray
import random

class StreamingAmbientDataset(StreamingDataset):
    """Dataset class for loading precomputed latents from mds format.
    
    Args:
        streams: List of individual streams (in our case streams of individual datasets)
        shuffle: Whether to shuffle the dataset
        image_size: Size of images (256 or 512)
        cap_seq_size: Context length of text-encoder
        cap_emb_dim: Dimension of caption embeddings
        cap_drop_prob: Probability of using all zeros caption embedding (classifier-free guidance)
        batch_size: Batch size for streaming
    """

    def __init__(
        self,
        # Micro diffusion params
        streams: Optional[List[Stream]] = None,
        shuffle: bool = False,
        image_size: Optional[int] = None,
        cap_seq_size: Optional[int] = None,
        cap_emb_dim: Optional[int] = None,
        cap_drop_prob: float = 0.0,
        batch_size: Optional[int] = None,
        # EDM params
        p_mean: float = -0.6,
        p_std: float = 1.2,
        # Ambient params
        sa1b_sigma: float = 0.0,
        cc12m_sigma: float = 0.0,
        textcaps_sigma: float = 0.0,
        jdb_sigma: float = 0.0,
        diffdb_sigma: float = 0.0,
        ambient_buffer: float = 0.05,
        **kwargs
    ) -> None:
        super().__init__(
            streams=streams,
            shuffle=shuffle,
            batch_size=batch_size,
        )

        # Micro diffusion config
        self.image_size = image_size
        self.cap_seq_size = cap_seq_size
        self.cap_emb_dim = cap_emb_dim
        self.cap_drop_prob = cap_drop_prob

        # EDM config
        self.edm_config = EasyDict({
            'sigma_min': 0.002,
            'sigma_max': 80,
            'P_mean': p_mean,
            'P_std': p_std,
            'sigma_data': 0.9,
            'num_steps': 18,
            'rho': 7,
            'S_churn': 0,
            'S_min': 0,
            'S_max': float('inf'),
            'S_noise': 1
        })

        # Ambient config
        self.ambient_config = EasyDict({
            'sa1b_sigma':sa1b_sigma,
            'cc12m_sigma':cc12m_sigma,
            'textcaps_sigma':textcaps_sigma,
            'jdb_sigma':jdb_sigma,
            'diffdb_sigma':diffdb_sigma,
            'buffer':ambient_buffer,
        })

    def get_sigma_tn(self, index: int) -> float:
        shard_id, shard_sample_id = self.spanner[index]
        shard = self.shards[shard_id]
        sigma_tn = 0.0
        for dirname in ['sa1b', 'cc12m', 'textcaps', 'jdb', 'diffdb']:
            if f'/{dirname}/' in shard.dirname:
                sigma_tn = self.ambient_config[f'{dirname}_sigma']            
        return sigma_tn

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, str, float]]:
        sample = super().__getitem__(index)
        out = {}

        # Micro diffusion
        out['drop_caption_mask'] = (
            0. if torch.rand(1) < self.cap_drop_prob else 1.
        )
        out['caption_latents'] = torch.from_numpy(
            np.frombuffer(sample['caption_latents'], dtype=np.float16)
            .copy()
        ).reshape(1, self.cap_seq_size, self.cap_emb_dim)

        if self.image_size == 256 and 'latents_256' in sample:
            out['image_latents'] = torch.from_numpy(
                np.frombuffer(sample['latents_256'], dtype=np.float16)
                .copy()
            ).reshape(-1, 32, 32)

        if self.image_size == 512 and 'latents_512' in sample:
            out['image_latents'] = torch.from_numpy(
                np.frombuffer(sample['latents_512'], dtype=np.float16)
                .copy()
            ).reshape(-1, 64, 64)
        
        # Ambient
        device = out['image_latents'].device
        dtype = out['image_latents'].dtype
        ## Noise
        g_gpu = torch.Generator(device=device)
        g_gpu.manual_seed(int(index))
        noise = torch.empty_like(out['image_latents']).normal_(generator=g_gpu)
        out['noise'] = noise
        ## Sigma_tn
        sigma_tn = self.get_sigma_tn(index)
        out['sigma_tn'] = torch.Tensor([sigma_tn]).to(device)

        return out

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over all the samples in our partition with rejection sampling.
        
        Returns:
            Iterator[Dict[str, Any]]: Each accepted sample.
        """
        # Exit the threads that are pre-downloading and iterating the shards for previous epoch
        if hasattr(self, '_iterator'):
            self._iterator.exit()

        # For exception handling
        if not hasattr(self, '_executor'):
            self._executor = ThreadPoolExecutor()
        if not hasattr(self, '_event'):
            self._event = Event()
        elif self._event.is_set():
            raise RuntimeError('Background thread failed. Check other traceback.')

        # Discover where we left off, or start at the next epoch
        self._unique_worker_world = self._unique_rank_world.detect_workers()
        self._parallel_worker_world = self._parallel_rank_world.detect_workers()
        epoch, sample_in_epoch = self._resume_incr_epoch()

        # Get this worker's partition of samples to process
        sample_ids = self._get_work(epoch, sample_in_epoch)
        if not len(sample_ids):  # Resumed at end of epoch, out of samples
            return

        # Iterate over the samples while downloading ahead
        self._iterator = it = _Iterator(sample_ids)
        prepare_future = self._executor.submit(self._prepare_thread, it)
        prepare_future.add_done_callback(self.on_exception)
        ready_future = self._executor.submit(self._ready_thread, it)
        ready_future.add_done_callback(self.on_exception)
        
        # Generate sigma_t before sample
        rnd_normal = torch.randn((1,))
        sigma_t = (rnd_normal * self.edm_config.P_std + self.edm_config.P_mean).exp()

        # Apply rejection sampling logic
        for sample_id in self._each_sample_id(it):
            sigma_tn = self.get_sigma_tn(sample_id)
            
            # Rejection sampling
            if (sigma_t > sigma_tn + self.ambient_config.buffer) or (sigma_tn == 0):
                sample = self.__getitem__(sample_id)
                sample['sigma_t'] = sigma_t
                yield sample

                # Get next sigma_t
                rnd_normal = torch.randn((1,))
                sigma_t = (rnd_normal * self.edm_config.P_std + self.edm_config.P_mean).exp()
        
        wait([prepare_future, ready_future], return_when='FIRST_EXCEPTION')
        it.exit()

def build_streaming_latents_dataloader(
    datadir: Union[str, List[str]],
    batch_size: int,
    image_size: int = 256,
    cap_seq_size: int = 77,
    cap_emb_dim: int = 1024,
    cap_drop_prob: float = 0.0,
    shuffle: bool = True,
    drop_last: bool = True,
    # EDM params
    p_mean: float = -0.6,
    p_std: float = 1.2,
    # Ambient params
    sa1b_sigma: float = 0.0,
    cc12m_sigma: float = 0.0,
    textcaps_sigma: float = 0.0,
    jdb_sigma: float = 0.0,
    diffdb_sigma: float = 0.0,
    ambient_buffer: float = 0.05,
    **dataloader_kwargs
) -> DataLoader:
    """Creates a DataLoader for streaming latents dataset."""
    if isinstance(datadir, str):
        datadir = [datadir]

    streams = [Stream(remote=None, local=d) for d in datadir]

    dataset = StreamingAmbientDataset(
        streams=streams,
        shuffle=shuffle,
        image_size=image_size,
        cap_seq_size=cap_seq_size,
        cap_emb_dim=cap_emb_dim,
        cap_drop_prob=cap_drop_prob,
        batch_size=batch_size,
        # EDM params
        p_mean=p_mean,
        p_std=p_std,
        # Ambient params
        sa1b_sigma=sa1b_sigma,
        cc12m_sigma=cc12m_sigma,
        textcaps_sigma=textcaps_sigma,
        jdb_sigma=jdb_sigma,
        diffdb_sigma=diffdb_sigma,
        ambient_buffer=ambient_buffer,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=None,
        drop_last=drop_last,
        **dataloader_kwargs,
    )

    return dataloader
