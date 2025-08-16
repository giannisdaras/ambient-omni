
# Ambient + Latent Diffusion w/ EDM2

This part of the repo focuses on large-scale latent diffusion experiments with [EDM2](https://github.com/NVlabs/edm2) using ImageNet. Here, we show how just by using the data better, you can improve the performance and quality of your generative models. Everything else, including data, model size, and training FLOPs, stays the exact same. The key idea is to change the diffusion times we use for our images depending on quality: high-quality data is used for all times, while low-quality data is used only for times above a certain threshold.

You can find our best model from the paper on [huggingface](https://huggingface.co/giannisdaras/ambient-o-imagenet512-xxl-with-crops), and if you want to train your own, only three things are required:
1. Prepare your ImageNet data in the EMD2 format
2. Calculate the quality of your data. In this repo, we use CLIP-IQA, but any metric could work.
3. Train your generative model using the ambient-o algorithm.

## 1. Prepare your data in the EMD2 format

Follow the [Preparing datasets](https://github.com/NVlabs/edm2#preparing-datasets) section in the original EDM-2 repository.

## 2. Calculate the quality of your data

Just run `scripts/calculate_metrics_quality.sh` or use our annotations uploaded to [huggingface](https://huggingface.co/datasets/adrianrm/ambient-o-clip-iqa-patches-imagenet). Our sample training scripts are already set-up to use the huggingface data, so you don't have to do anything. If you want to load them outside the trainign code for analysis, you can do so like this:

```
from huggingface_hub import hf_hub_download

annotations_qualities_path = hf_hub_download(repo_id='adrianrm/ambient-o-clip-iqa-patches-imagenet', filename="clip_iqa_patch_average.safetensors", repo_type="dataset")
annotations_qualities = {}
with safe_open(annotations_qualities_path, framework="pt", device=dist.get_rank()) as f:
    for k in f.keys():
        annotations_qualities[k] = f.get_tensor(k)
```


## 3. Train your diffusion model

We provide scripts for training our XXL ambient models with crops (`scripts/train_edm2_ambient-o_xxl.sh`) and without crops (`scripts/train_edm2_ambient-o_crops_xxl.sh`). While the scripts are set up to run on a single 8 GPU node for simplicity, we suggest following the original EDM2 recommendation of running it on at least 4 such nodes (32 GPUs total).

## Evaluation

We provide scripts for evaluating our strongest model, published on huggingface, in the scripts `scripts/eval_edm2_ambient-o_crops_xxl_dino_FD.sh` and `scripts/eval_edm2_ambient-o_crops_xxl_FID.sh`. They can easily be adapted to your own models by replacing the `experiment_dir` variable to your own experiment directory, uncommenting the part of the code that generates the post-hoc EMAs, and setting the value of `net` to the EMA path instead of our huggingface repo. For any doubts, refer to the original EDM2 repository.

# 🔗 Related Codebases

* [EDM2](https://github.com/NVlabs/edm2): starting point for this repository.
* [Ambient utils](https://github.com/giannisdaras/ambient-utils): helper functions for training diffusion models (or flow matching models) in settings with limited access to high-quality data.
* [Ambient Laws](https://github.com/giannisdaras/ambient-laws): trains models with a mix of clean and noisy data.
* [Ambient Diffusion](https://github.com/giannisdaras/ambient-diffusion): trains models for linear corruptions.
* [Consistent Diffusion Meets Tweedie](https://github.com/giannisdaras/ambient-tweedie): trains models with only noisy data, with support for Stable Diffusion finetuning.
* [Consistent Diffusion Models](https://github.com/giannisdaras/cdm): original implementation of the consistency loss.


# 📧 Contact

If you are interested in colaborating, please reach out to gdaras[at]mit[dot]edu.
