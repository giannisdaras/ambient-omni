from setuptools import setup

setup(
    name="micro_diffusion",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'accelerate',
        'diffusers',
        'huggingface_hub',
        'torch',
        'torchvision',
        'transformers',
        'timm',
        'open_clip_torch<=2.24.0',
        'easydict',
        'einops',
        'mosaicml-streaming',
        'torchmetrics',
        'mosaicml[tensorboard, wandb]',
        'tqdm',
        'pandas',
        'fastparquet',
        'omegaconf', 
        'datasets', 
        'hydra-core',
        'beautifulsoup4',
        'ambient-utils',
        'safetensors',
        'clean-fid'
    ],
)
