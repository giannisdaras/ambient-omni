import time
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from composer.core import Precision
from composer.utils import dist, reproducibility
from composer.algorithms import GradientClipping
from composer.algorithms.low_precision_layernorm import apply_low_precision_layernorm
from micro_diffusion.models.utils import text_encoder_embedding_format
import torch.distributed as dist
torch.backends.cudnn.benchmark = True  # 3-5% speedup
from ambient_utils import dist_utils
import os
from composer.algorithms import EMA

@hydra.main(version_base=None)
def train(cfg: DictConfig) -> None:
    """Train a micro-diffusion model using the provided configuration.

    Args:
        cfg (DictConfig): Configuration object loaded from yaml file.
    """
    # Set NODE_RANK for distributed training
    NODE_RANK = int(os.environ.get('RANK', 0))
    print(f"Setting NODE_RANK to {NODE_RANK}")
    os.environ['NODE_RANK'] = str(NODE_RANK)
    print(f"NODE_RANK: {os.environ['NODE_RANK']}")

    if not cfg:
        raise ValueError('Config not specified. Please provide --config-path and --config-name, respectively.')
    reproducibility.seed_all(cfg['seed'])

    assert cfg.model.precomputed_latents, "For microbudget training, we assume that latents are already precomputed for all datasets"
    print("Instantiating model")
    model = hydra.utils.instantiate(cfg.model)
    print("Model instantiated")

    # Set up optimizer with special handling for MoE parameters
    print("Starting optimizer setup...")
    moe_params = [p[1] for p in model.dit.named_parameters() if 'moe' in p[0].lower()]
    rest_params = [p[1] for p in model.dit.named_parameters() if 'moe' not in p[0].lower()]
    print(f"Separated MoE ({len(moe_params)}) and non-MoE ({len(rest_params)}) parameters.")

    if len(moe_params) > 0:
        print('Reducing learning rate of MoE parameters by 1/2')
        opt_dict = dict(cfg.optimizer)
        opt_name = opt_dict['_target_'].split('.')[-1]
        del opt_dict['_target_']
        optimizer = getattr(torch.optim, opt_name)(
            params=[{'params': rest_params}, {'params': moe_params, 'lr': cfg.optimizer.lr / 2}], **opt_dict)
    else:
        print("Instantiating optimizer with hydra...")
        optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.dit.parameters())
    
    print("Optimizer instantiated.")

    # Convert ListConfig betas to native list to avoid ValueError when saving optimizer state
    print("Fixing optimizer betas...")
    for p in optimizer.param_groups:
        p['betas'] = list(p['betas'])
    print("Optimizer betas fixed.")

    # Set up data loaders
    print("Starting data loader setup...")
    cap_seq_size, cap_emb_dim = text_encoder_embedding_format(cfg.model.text_encoder_name)
    print("Setting barrier before instantiating train_loader")
    # torch.distributed.barrier()
    print("Instantiating train_loader")
    
    train_loader = hydra.utils.instantiate(
        cfg.dataset.train,
        image_size=cfg.dataset.image_size,
        batch_size=cfg.dataset.train_batch_size // dist.get_world_size(),
        cap_seq_size=cap_seq_size,
        cap_emb_dim=cap_emb_dim,
        cap_drop_prob=cfg.dataset.cap_drop_prob)

    model.train_loader = train_loader
    # sample_batch = next(iter(train_loader))
    # print(f"Rank {dist.get_rank()}: index tensor: {sample_batch['index'][:4]}")
    # print(f"Rank {dist.get_rank()}: shard_id tensor: {sample_batch['shard_id'][:4]}")
    # print(f"Rank {dist.get_rank()}: shard_sample_id tensor: {sample_batch['shard_sample_id'][:4]}")
    
    print(f"Found {len(train_loader.dataset)*dist.get_world_size()} images in the training dataset")
    # print("Setting barrier before instantiating eval_loader")

    # torch.distributed.barrier()
    print("Instantiating eval_loader")
    eval_loader = hydra.utils.instantiate(
        cfg.dataset.eval,
        image_size=cfg.dataset.image_size,
        batch_size=cfg.dataset.eval_batch_size // dist.get_world_size(),
        cap_seq_size=cap_seq_size,
        cap_emb_dim=cap_emb_dim)

    # torch.distributed.barrier()
    print("Eval loader instantiated")

    # Initialize training components
    logger, callbacks, algorithms = [], [], []

    # Set up loggers
    for log, log_conf in cfg.logger.items():
        if '_target_' in log_conf:
            if log == 'wandb':
                wandb_logger = hydra.utils.instantiate(log_conf, _partial_=True)
                logger.append(wandb_logger(init_kwargs={'config': OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)}))
            else:
                logger.append(hydra.utils.instantiate(log_conf))

    # Configure algorithms
    if 'algorithms' in cfg:
        for alg_name, alg_conf in cfg.algorithms.items():
            if alg_name == 'low_precision_layernorm':
                apply_low_precision_layernorm(model=model.dit,
                                              precision=Precision(alg_conf['precision']),
                                              optimizers=optimizer)
            elif alg_name == 'gradient_clipping':
                algorithms.append(GradientClipping(clipping_type='norm', clipping_threshold=alg_conf['clip_norm']))
            elif alg_name == 'ema':
                ema_config = hydra.utils.instantiate(alg_conf)
                ema_config['half_life'] = None
                ema = EMA(**ema_config)
                algorithms.append(ema)
                print(f'Algorithm {alg_name} instantiated')
            else:
                print(f'Algorithm {alg_name} not supported.')

    # Set up callbacks
    if 'callbacks' in cfg:
        for _, call_conf in cfg.callbacks.items():
            if '_target_' in call_conf:
                print(f'Instantiating callbacks: {call_conf._target_}')
                callbacks.append(hydra.utils.instantiate(call_conf))

    scheduler = hydra.utils.instantiate(cfg.scheduler)

    # disable online evals if using torch.compile
    if cfg.misc.compile:
        cfg.trainer.eval_interval = 0

    # torch.distributed.barrier()
    
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        optimizers=optimizer,
        model=model,
        loggers=logger,
        algorithms=algorithms,
        schedulers=scheduler,
        callbacks=callbacks,
        precision='amp_bf16' if cfg.model['dtype'] == 'bfloat16' else 'amp_fp16',  # fp16 by default
        python_log_level='debug',
        compile_config={} if cfg.misc.compile else None  # it enables torch.compile (~15% speedup)
    )
    print("Trainer instantiated")
    # Ensure models are on correct device
    device = next(model.dit.parameters()).device
    model.vae.to(device)
    model.text_encoder.to(device)
    print("Models moved to device")
    print("About to start training with trainer.fit()")

    return trainer.fit()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    dist_utils.init()
    # Explicitly set device ID for each process to avoid ProcessGroupNCCL warnings
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    # Force PyTorch to use this device for barrier operations
    if torch.distributed.is_initialized():
        torch.distributed.barrier(device_ids=[local_rank])
    dist_utils.print0("Total nodes: ", dist_utils.get_world_size())
    train()
