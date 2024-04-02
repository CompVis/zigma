from collections import OrderedDict
import os
import glob
import numpy as np
import torch
from PIL import Image
import logging
import importlib
import logging


def wandb_runid_from_checkpoint(checkpoint_path):
    # Python
    import os
    import re

    # Define the directory to search
    dir_path = os.path.join(checkpoint_path, "wandb/latest-run")
    # Define the pattern to match
    pattern = r"run-(\w+)\.wandb"

    # Iterate over all files in the directory
    for filename in os.listdir(dir_path):
        # If the filename matches the pattern
        if re.match(pattern, filename):
            # Extract the desired part of the filename
            extracted_part = re.match(pattern, filename).group(1)
            logging.info(f"induced run id from ckpt: {extracted_part}")
            return extracted_part
    raise ValueError("No file found that matches the pattern")


def instantiate_from_config(config):
    module_name, class_name = config["target"].rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    instance = class_(**config.get("params", {}))
    return instance


def get_model(args, device):
    if args.is_latent:
        # Create model:
        assert (
            args.data.image_size % 8 == 0
        ), "Image size must be divisible by 8 (for the VAE encoder)."
        in_channels = 4
        input_size = args.data.image_size // 8
    else:
        in_channels = 3
        input_size = args.data.image_size
    logging.info(f"input_size: {input_size}, in_channels: {in_channels}")

    model = instantiate_from_config(args.model).to(device)

    return model, in_channels, input_size


def create_logger(rank, logging_dir=None):
    """
    Create a logger that writes to a log file and stdout.
    """
    if rank == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ],
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def grad_clip(opt, model, max_grad_norm=2.0):
    if hasattr(opt, "clip_grad_norm"):
        # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
        opt.clip_grad_norm(max_grad_norm)
    else:
        # Revert to normal clipping otherwise, handling Apex or full precision
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),  # amp.master_params(self.opt) if self.use_apex else
            max_grad_norm,
        )


def get_latest_checkpoint(checkpoint_dir):

    # Get a list of all checkpoint files
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*"))
    # Check if there are any checkpoints
    if not checkpoint_files:
        # print("No checkpoints found")
        # raise FileNotFoundError
        return "No_checkpoints_found"
    else:
        # Get the checkpoint file with the latest modification time
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        # print(f"Latest checkpoint: {latest_checkpoint}")
        return latest_checkpoint
