import argparse
from ambient_utils import save_image
from ambient_utils import dist_utils
from ambient_utils.dataset import SyntheticallyCorruptedImageFolderDataset
from ambient_utils.classifier import get_classifier_trajectory
import torch
from torch_utils.misc import copy_params_and_buffers
import pickle
import dnnlib
import os
import json
import importlib
from slurm_jobs.utils import find_training_folders_based_on_params, find_nearest_checkpoint, find_latest_checkpoint
import matplotlib.pyplot as plt
from filelock import FileLock
from tqdm import tqdm
import datetime
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
parser = argparse.ArgumentParser()
parser.add_argument("--annotated_dataset_path", type=str, required=True, help="Path to save the annotated dataset.")
parser.add_argument("--checkpoint_path", type=str, required=True, help="Checkpoint path.")
parser.add_argument("--training_noise_config", type=str, default="blurs", help="Corruption configuration during the training of the classifier.")
parser.add_argument("--inference_noise_config", type=str, default="blurs1", help="Corruption configuration for the inference of the classifier.")
parser.add_argument("--dataset", type=str, default="cifar", help="Dataset name.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
parser.add_argument("--num_trials_per_t", type=int, default=4, help="How many times to query the classifier for each t.")
parser.add_argument("--num_sigmas", type=int, default=2048, help="How many sigmas for dense annotation")
parser.add_argument("--corruption_probability", type=float, default=0.5, help="Corruption probability.")
# TODO (@giannisdaras): update this path once we get a better one
parser.add_argument("--checkpoint_index", type=int, default=200_000, help="Checkpoint index.")

def load_net_from_pkl(ckpt_file):
    if ckpt_file.startswith('adrianrm/ambient-o'):
        options_path = hf_hub_download(repo_id=ckpt_file, filename="training_options.json")
        options = json.load(open(options_path, "r", encoding="utf-8"))

        interface_kwargs_path = hf_hub_download(repo_id=ckpt_file, filename="interface_kwargs.json")
        interface_kwargs = json.load(open(interface_kwargs_path, "r", encoding="utf-8"))

        net = dnnlib.util.construct_class_by_name(**options['network_kwargs'], **interface_kwargs)

        state_dict_path = hf_hub_download(repo_id=ckpt_file, filename="ema.safetensors")
        state_dict = load_file(state_dict_path)
        net.load_state_dict(state_dict)
        net.eval()

        # options["dataset_kwargs"]["path"] = your cifar10 path

        assert_msg = f"""
        HuggingFace checkpoint expects data in ./data/cifar10/train; if your data is in a different path, please create symlink or edit options["dataset_kwargs"]["path"] here.
        """
        assert os.path.exists('./data/cifar10/train'), assert_msg
    else:
        base_folder = os.path.dirname(ckpt_file)
        with open(os.path.join(base_folder, "training_options.json"), "r", encoding="utf-8") as f:
            options = json.load(f)

        interface_kwargs = dict(img_resolution=options['dataset_kwargs']['resolution'], img_channels=3, label_dim=0)
        
        net = dnnlib.util.construct_class_by_name(**options['network_kwargs'], **interface_kwargs)
        with dnnlib.util.open_url(ckpt_file) as f:
            data = pickle.load(f)
        copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
    return net, options




def main(args):
    torch.multiprocessing.set_start_method('spawn')
    dist_utils.init()
    
    dist_utils.print0("Total nodes: ", dist_utils.get_world_size())

    if args.checkpoint_path is None:
        training_folders = find_training_folders_based_on_params(corruption_probability=0.5, 
                                                                noise_config=args.training_noise_config, 
                                                                dataset_keep_percentage=1.0, 
                                                                dataset=args.dataset)        
        checkpoint_path = find_nearest_checkpoint(training_folders, checkpoint_index=args.checkpoint_index, pkl=True)    
    else:
        checkpoint_path = args.checkpoint_path

    net, options = load_net_from_pkl(checkpoint_path)
    net.eval().to("cuda")
    
    # prepare params for synthetic dataset corruption
    corruptions_dict = importlib.import_module(f"noise_configs.inference.{args.inference_noise_config}").corruptions_dict
    options['dataset_kwargs']['corruptions_dict'] = corruptions_dict
    del options['dataset_kwargs']['noise_config']
    del options['dataset_kwargs']['dataset_keep_percentage']  # TODO: ?
    # overwrite the corruption probability so that we can create different types of datasets, as needed.
    options['dataset_kwargs']['corruption_probability'] = args.corruption_probability


    dataset_obj = SyntheticallyCorruptedImageFolderDataset(**options['dataset_kwargs'])
    dataset_loader = torch.utils.data.DataLoader(
            dataset_obj,
            batch_size=1,
            shuffle=False,
            sampler=torch.utils.data.distributed.DistributedSampler(dataset_obj, shuffle=False)
            )

    # rnd_normal = torch.randn([4096, 1, 1, 1], device="cuda", generator=torch.Generator(device="cuda").manual_seed(42))  # it is very important to set this seed to have consistency with the seed used during training.
    rnd_normal = torch.randn([args.num_sigmas, 1, 1, 1], device="cuda", generator=torch.Generator(device="cuda").manual_seed(42))  # it is very important to set this seed to have consistency with the seed used during training.
    sigmas, _ = (rnd_normal * 1.2 - 1.2).exp().sort(dim=0)
    sigmas = sigmas.squeeze()

    if dist_utils.get_rank() == 0:
        os.makedirs(args.annotated_dataset_path, exist_ok=True)
        # dump the sigmas into a file
        with open(os.path.join(args.annotated_dataset_path, "sigmas.txt"), "w") as f:
            for sigma in sigmas:
                f.write(f"{sigma.item()}\n")
     
    def scheduler(x, s):
        return x + s * torch.randn_like(x)

    # Resume
    process_id = torch.distributed.get_rank()
    annotations_file = os.path.join(args.annotated_dataset_path, f"annotations_{process_id}.jsonl")
    annotations = {}
    if os.path.exists(annotations_file):
        with open(annotations_file, "r") as f:
            for line in f:
                line_json = json.loads(line)
                filename = line_json["filename"]
                annotations[filename] = True

    # all models wait for the path to be created
    torch.distributed.barrier()
    process_id = torch.distributed.get_rank()
    for dataset_item in tqdm(dataset_loader):
        images = dataset_item["image"].to("cuda").repeat(args.num_trials_per_t, 1, 1, 1)
        labels = dataset_item["label"].to("cuda").repeat(args.num_trials_per_t, 1)  # attention: this is NOT the label for good or bad image. This is more like a class label (dog, cat, etc.)

        image_name = dataset_item['filename'][0].split("/")[-1]
        if image_name not in annotations:
            image_path = os.path.join(args.annotated_dataset_path, image_name)
            save_image(images[0], image_path)


            def model_fn(x, s):
                return net(x, s, labels, augment_labels=None)["cls_logits"]


            # Write to process-specific annotation file
            process_id = torch.distributed.get_rank()
            process_annotations_path = os.path.join(args.annotated_dataset_path, f"annotations_{process_id}.jsonl")

            if dataset_item['corruption_label'].sum() == 0:
                sigma = 0.0
            else:
                probs = get_classifier_trajectory(
                    input=images,
                    scheduler=scheduler,
                    model=model_fn,
                    diffusion_times=sigmas,
                    batch_size=args.batch_size,
                    model_output_type='logits'
                )
                sigma = None
            
            with open(process_annotations_path, "a", encoding="utf-8") as f:
                if sigma is None:
                    annotation = {
                        "filename": image_name,
                        "probabilities": probs.tolist(),
                    }
                else:
                    annotation = {
                        "filename": image_name,
                        "annotation": sigma
                    }
                f.write(json.dumps(annotation) + "\n")
    
    # After all processes finish, merge files
    torch.distributed.barrier()
    # torch.distributed.monitored_barrier(timeout=datetime.timedelta(days=1))
    if process_id == 0:
        # Merge all process files into single annotations file
        final_annotations_path = os.path.join(args.annotated_dataset_path, "annotations.jsonl")
        with open(final_annotations_path, "a+", encoding="utf-8") as outfile:
            world_size = torch.distributed.get_world_size()
            for pid in range(world_size):
                proc_file = os.path.join(args.annotated_dataset_path, f"annotations_{pid}.jsonl")
                if os.path.exists(proc_file):
                    with open(proc_file, encoding='utf-8') as infile:
                        outfile.write(infile.read())
                    os.remove(proc_file)  # Clean up process file
        


        # # plt.figure()
        # # plt.plot(probs)
        # # plt.savefig(f"probs_{dataset_item['filename'][0].split('/')[-1].split('.')[0]}.pdf")

        # # save_images(images, f"images_{dataset_item['filename'][0].split('/')[-1].split('.')[0]}.png")
        # import pdb; pdb.set_trace()





if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
