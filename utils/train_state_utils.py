import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
#from torchvision.utils import save_image
import logging


raise NotImplementedError("This file is not used in the project")


def ema(model_dest: nn.Module, model_src: nn.Module, rate):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)


class TrainState(object):
    def __init__(self, optimizer, step, model=None, model_ema=None):
        self.optimizer = optimizer
        self.step = step
        self.model = model
        self.model_ema = model_ema

    def ema_update(self, rate=0.9999):
        if self.model_ema is not None:
            ema(self.model_ema, self.model, rate)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.step, os.path.join(path, "step.pth"))
        for key, val in self.__dict__.items():
            if key != "step" and val is not None:
                torch.save(val.state_dict(), os.path.join(path, f"{key}.pth"))

    def load(self, path):
        logging.info(f"load from {path}")
        self.step = torch.load(os.path.join(path, "step.pth"))
        for key, val in self.__dict__.items():
            if key != "step" and val is not None:
                val.load_state_dict(
                    torch.load(os.path.join(path, f"{key}.pth"), map_location="cpu")
                )

    def resume(self, ckpt_root, step=None):
        if not os.path.exists(ckpt_root):
            return
        if step is None:
            ckpts = list(filter(lambda x: ".ckpt" in x, os.listdir(ckpt_root)))
            if not ckpts:
                return
            steps = map(lambda x: int(x.split(".")[0]), ckpts)
            step = max(steps)
        ckpt_path = os.path.join(ckpt_root, f"{step}.ckpt")
        logging.info(f"resume from {ckpt_path}")
        self.load(ckpt_path)

    def to(self, device):
        for key, val in self.__dict__.items():
            if isinstance(val, nn.Module):
                val.to(device)


def cnt_params(model):
    return sum(param.numel() for param in model.parameters())


def initialize_train_state(config, model, model_ema , device):
    params = []
    params += model.parameters()
    model_ema.eval()
    logging.warning(f"nnet has {cnt_params(model)} parameters")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.optim.lr, weight_decay=config.optim.wd
    )
  
    train_state = TrainState(
        optimizer=optimizer,
        step=0,
        model=model,
        model_ema=model_ema,
    )
    train_state.ema_update(0)
    if device is not None:
        train_state.to(device)
    return train_state