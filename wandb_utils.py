import numpy as np
import wandb
import torch
from torchvision.utils import make_grid
import torch.distributed as dist
from PIL import Image
import os
import argparse
import hashlib
import math
from omegaconf import OmegaConf
import os

import re

from datasets.wds_dataloader import WebDataModuleFromConfig
from model_zigma import get_2d_sincos_pos_embed


def is_main_process():
    return dist.get_rank() == 0


def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
        for k, v in vars(namespace).items()
    }


def get_max_ckpt_from_dir(dir_path):
    dir_path = os.path.join(dir_path, "checkpoints")
    # Define the pattern to match
    pattern = r"(\d+)\.pt"

    # Initialize the maximum step number and corresponding file name
    max_step = -1
    max_step_file = None

    # Iterate over all files in the directory
    for filename in os.listdir(dir_path):
        # If the filename matches the pattern
        match = re.match(pattern, filename)
        if match:
            # Extract the step number from the filename
            step = int(match.group(1))
            # If this step number is larger than the current maximum
            if step > max_step:
                # Update the maximum step number and corresponding file name
                max_step = step
                max_step_file = filename

    if max_step_file is None:
        raise ValueError(f"No checkpoint files found in {dir_path}")
    else:
        print(
            f"Found checkpoint file {max_step_file} with step {max_step} from {dir_path}"
        )
        return os.path.join(dir_path, max_step_file)


def generate_run_id(exp_name):
    # https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
    return str(int(hashlib.sha256(exp_name.encode("utf-8")).hexdigest(), 16) % 10**8)


def initialize(args, entity, exp_name, project_name, wandb_dir):
    config_dict = OmegaConf.to_container(args, resolve=True)
    # wandb.login(key=args.wandb.key)
    wandb.init(
        project=project_name,
        name=exp_name,
        config=config_dict,
        dir=wandb_dir,
        resume="allow",
        mode="online",
    )


def log(stats, step=None):
    if is_main_process():
        wandb.log({k: v for k, v in stats.items()}, step=step)


def log_image(sample, step=None):
    if is_main_process():
        sample = array2grid(sample, to255=True)
        wandb.log({f"samples": wandb.Image(sample), "train_step": step})


def array2grid(x, to255=False):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x, nrow=nrow, normalize=True, value_range=(-1, 1))
    if to255:
        x = (
            x.mul(255)
            .add_(0.5)
            .clamp_(0, 255)
            .permute(1, 2, 0)
            .to("cpu", torch.uint8)
            .numpy()
        )
    else:
        x = x.permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return x


def array2grid_pixel(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x, nrow=nrow, normalize=False)
    x = x.permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return x


def test_sd_1_5():
    from diffusers import StableDiffusionPipeline
    import torch

    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    vae = pipe.vae.to("cuda")
    img = torch.randn(4, 3, 256, 256).to("cuda")
    with torch.no_grad():
        latents = vae.encode(img).latent_dist.sample()
    print(latents.shape)


def test_webdataset_faceshq_sd1_5():
    from omegaconf import OmegaConf

    config = OmegaConf.load("datasets/debug_webface.yaml")
    datamod = WebDataModuleFromConfig(**config["data"]["params"])
    # from pudb import set_trace; set_trace()
    dataloader = datamod.train_dataloader()
    if True:
        from diffusers import StableDiffusionPipeline

        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(model_id)
        vae = pipe.vae.to("cuda")

    for i, batch in enumerate(dataloader):
        print(batch.keys())
        print(
            batch["image"].shape,
            f"max: {batch['image'].max()}, min: {batch['image'].min()}",
        )
        print(f"Batch number: {i}")
        print(batch["latent"].shape)
        for _ in range(len(batch["latent"])):
            _img, _latent = batch["image"][_].to("cuda"), batch["latent"][_].to("cuda")
            print(_img.shape, _latent.shape)
            img_recovered = vae.decode(_latent.unsqueeze(0)).sample[0]
            print(img_recovered.shape)
            _img = (_img * 0.5 + 0.5).cpu()
            img_recovered = (img_recovered * 0.5 + 0.5).cpu().detach()
            img = Image.fromarray((_img.permute(1, 2, 0).numpy() * 255).astype("uint8"))
            img_recovered = Image.fromarray(
                (img_recovered.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            )
            img.save(f"img_{_}.png")
            img_recovered.save(f"img_recovered_{_}.png")

        break
    print("end")


import matplotlib.pyplot as plt


def vis_position_embedding(grid_size=33, dim=512):
    if False:
        ref_x = grid_size // 2
        ref_y = grid_size // 2
    elif True:
        ref_x = 0
        ref_y = 0
    pos_embed = get_2d_sincos_pos_embed(dim, grid_size)
    pos_embed_3d = pos_embed.reshape(dim, grid_size, grid_size)
    reference_pts = pos_embed_3d[:, ref_x : ref_x + 1, ref_y : ref_y + 1]
    distance = np.linalg.norm(pos_embed_3d - reference_pts, ord=1, axis=0)
    print(distance.shape)
    plt.imshow(distance, cmap="inferno")
    plt.colorbar()
    plt.savefig("distance_pe_vis.png")
    plt.show()


if __name__ == "__main__":
    # test_sd_1_5()
    # test_webdataset_faceshq_sd1_5()
    vis_position_embedding()
    print("aa")
