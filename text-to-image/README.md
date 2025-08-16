
# Ambient + Text-to-Image Diffusion w/ Micro-Diffusion

This part of the repo focuses on large-scale text-to-image diffusion experiments with [Micro-Diffusion](https://github.com/SonyResearch/micro_diffusion) using Conceptual Captions, Segment Anything-1B, TextCaps, JourneyDB, and DiffusionDB. Here, we show how just by using the data better, you can improve the performance and quality of your generative models. Everything else, including data, model size, and training FLOPs, stays the exact same. The key idea is to change the diffusion times we use for our images depending on quality: high-quality data is used for all times, while low-quality data is used only for times above a certain threshold.

You can find our best model from the paper on [huggingface](https://huggingface.co/giannisdaras/ambient-o), and if you want to train your own, only three things are required:
1. Prepare your environment and data following the instructions in the Micro-Diffusion repo
2. Train your generative model using the ambient-o algorithm.

## 1. Prepare your environment and data in the Micro-Diffusion format

Follow the instructions in the original [Micro-Diffusion](https://github.com/SonyResearch/micro_diffusion) repository (but using our edited version of their code).

## 2. Train your diffusion model

We provide scripts for training our ambient (`train_e2e_ambient.sh`) and baseline models (`train_e2e_baseline.sh`).

## Evaluation

We provide scripts for generating images for the COCO-30K (`scripts/generate_coco.sh`), drawbench (`scripts/generate_drawbench.sh`), and partiprompts (`scripts/generate_parti.sh`) benchmarks. We also provide scripts for evaluating FID (`scripts/eval_fid.sh`), CLIP-FD (`scripts/eval_clip-fd.sh`), CLIP alignment and quality (`scripts/eval_clip-align-and-quality.sh`), and GPT-4o (`scripts/eval_gpt4o.sh`) evaluations. The generation scripts use our [huggingface ambient checkpoint](https://huggingface.co/giannisdaras/ambient-o) by default, but you can change the path to your own models.

# 🔗 Related Codebases

* [Micro-Diffusion](https://github.com/SonyResearch/micro_diffusion): starting point for this repository.
* [Ambient utils](https://github.com/giannisdaras/ambient-utils): helper functions for training diffusion models (or flow matching models) in settings with limited access to high-quality data.
* [Ambient Laws](https://github.com/giannisdaras/ambient-laws): trains models with a mix of clean and noisy data.
* [Ambient Diffusion](https://github.com/giannisdaras/ambient-diffusion): trains models for linear corruptions.
* [Consistent Diffusion Meets Tweedie](https://github.com/giannisdaras/ambient-tweedie): trains models with only noisy data, with support for Stable Diffusion finetuning.
* [Consistent Diffusion Models](https://github.com/giannisdaras/cdm): original implementation of the consistency loss.


# 📧 Contact

If you are interested in colaborating, please reach out to gdaras[at]mit[dot]edu.
