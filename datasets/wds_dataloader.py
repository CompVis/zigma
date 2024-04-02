import os
import random
import torch
import numpy as np
import torchvision
import webdataset as wds
from einops import rearrange
import pytorch_lightning as pl
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

""" WebDataset """


def dict_collation_fn(samples, combine_tensors=True, combine_scalars=True):
    """Take a list  of samples (as dictionary) and create a batch, preserving the keys.
    If `tensors` is True, `ndarray` objects are combined into
    tensor batches.
    :param dict samples: list of samples
    :param bool tensors: whether to turn lists of ndarrays into a single ndarray
    :returns: single sample consisting of a batch
    :rtype: dict
    """
    keys = set.intersection(*[set(sample.keys()) for sample in samples])
    batched = {key: [] for key in keys}

    for s in samples:
        [batched[key].append(s[key]) for key in batched]

    result = {}
    for key in batched:
        if isinstance(batched[key][0], (int, float)):
            if combine_scalars:
                result[key] = np.array(list(batched[key]))
        elif isinstance(batched[key][0], torch.Tensor):
            if combine_tensors:
                result[key] = torch.stack(list(batched[key]))
        elif isinstance(batched[key][0], np.ndarray):
            if combine_tensors:
                result[key] = np.array(list(batched[key]))
        else:
            result[key] = list(batched[key])
    return result


class WebDataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self,
        tar_base,
        batch_size,
        image_size,
        train=None,
        validation=None,
        test=None,
        num_workers=4,
        multinode=True,
        is_video=False,
        min_size=None,
        max_pwatermark=1.0,
        video_frames=0,
        channel_last=False,
        val_batch_size=None,
        val_num_workers=None,
        **kwargs,
    ):
        super().__init__()
        print(f"Setting tar base to {tar_base}")
        self.tar_base = tar_base
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.train = train
        self.validation = validation
        self.test = test
        self.video_frames = video_frames
        self.multinode = multinode
        self.min_size = min_size  # filter out very small images
        self.max_pwatermark = max_pwatermark  # filter out watermarked images
        self.channel_last = channel_last
        self.val_batch_size = (
            val_batch_size if val_batch_size is not None else batch_size
        )
        self.val_num_workers = (
            val_num_workers if val_num_workers is not None else num_workers
        )
        self.is_video = is_video

    def make_loader(self, dataset_config, train=True):
        # change range from [0,1] to [-1,1] and put channel last or first
        image_transforms = []
        if self.channel_last:
            lambda_fn = lambda x: rearrange(x * 2.0 - 1.0, "c h w -> h w c")
        else:
            lambda_fn = lambda x: x * 2.0 - 1.0

        image_transforms.extend(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(self.image_size, antialias=True),
                torchvision.transforms.ConvertImageDtype(torch.float32),
                torchvision.transforms.Lambda(lambda_fn),
            ]
        )

        if "image_transforms" in dataset_config:
            image_transforms.extend(
                [instantiate_from_config(tt) for tt in dataset_config.image_transforms]
            )
        image_transforms = torchvision.transforms.Compose(image_transforms)

        if "transforms" in dataset_config:
            transforms_config = OmegaConf.to_container(dataset_config.transforms)
        else:
            transforms_config = dict()

        transform_dict = {
            dkey: (
                load_partial_from_config(transforms_config[dkey])
                if transforms_config[dkey] != "identity"
                else identity
            )
            for dkey in transforms_config
        }
        # this is crucial to set correct image key to get the transofrms applied correctly
        img_key = dataset_config.get("image_key", "image.png")
        transform_dict.update({img_key: image_transforms})

        if "postprocess" in dataset_config:
            postprocess = instantiate_from_config(dataset_config["postprocess"])
        else:
            postprocess = None
            print("No postprocess")

        # some illustration how shuffle works in webdataset
        # https://github.com/webdataset/webdataset/issues/71
        # TL;DR: set the shuffle as big as you can afford ->len(files)
        shuffle = dataset_config.get("shuffle", 0)
        shardshuffle = shuffle > 0

        nodesplitter = (
            wds.shardlists.split_by_node
            if self.multinode
            else wds.shardlists.single_node_only
        )

        tars = os.path.join(self.tar_base, dataset_config.shards)

        dset = (
            wds.WebDataset(
                tars,
                nodesplitter=nodesplitter,
                shardshuffle=shardshuffle,
                handler=wds.warn_and_continue,
            )
            .repeat()
            .shuffle(shuffle)
        )
        print(f"Loading webdataset with {len(dset.pipeline[0].urls)} shards.")

        if self.is_video:
            dset = dset.decode("rgb", handler=wds.warn_and_continue)
        else:
            dset = dset.decode("rgb", handler=wds.warn_and_continue).map_dict(
                **transform_dict, handler=wds.warn_and_continue
            )

        # change name of image key to be consistent with other datasets
        renaming = dataset_config.get("rename", None)
        if renaming is not None:
            dset = dset.rename(**renaming)

        if postprocess is not None:
            dset = dset.map(postprocess)

        bs = self.batch_size if train else self.val_batch_size
        nw = self.num_workers if train else self.val_num_workers
        dset = dset.batched(bs, partial=False, collation_fn=dict_collation_fn)
        loader = wds.WebLoader(dset, batch_size=None, shuffle=False, num_workers=nw)

        return loader

    def train_dataloader(self):
        return self.make_loader(self.train)

    def val_dataloader(self):
        return self.make_loader(self.validation, train=False)

    def test_dataloader(self):
        return self.make_loader(self.test, train=False)


if __name__ == "__main__":
    if False:
        from omegaconf import OmegaConf

        config = OmegaConf.load("debug_cifar10.yaml")
        datamod = WebDataModuleFromConfig(**config["data"]["params"])
        # from pudb import set_trace; set_trace()
        dataloader = datamod.train_dataloader()

        for i, batch in enumerate(dataloader):
            print(batch.keys())
            print(
                batch["image"].shape,
                f"max: {batch['image'].max()}, min: {batch['image'].min()}",
            )
            print(f"Batch number: {i}")
            break
        print("end")
    elif False:
        from omegaconf import OmegaConf

        config = OmegaConf.load("debug_celeba.yaml")
        datamod = WebDataModuleFromConfig(**config["data"]["params"])
        # from pudb import set_trace; set_trace()
        dataloader = datamod.train_dataloader()

        for i, batch in enumerate(dataloader):
            print(batch.keys())
            print(
                batch["image"].shape,
                f"max: {batch['image'].max()}, min: {batch['image'].min()}",
            )
            print("embedding", batch["embeddings"].shape)
            print("features", batch["features"].shape)
            print(f"Batch number: {i}")
            break
        print("end")
    elif False:
        from omegaconf import OmegaConf

        config = OmegaConf.load("debug_mm.yaml")
        datamod = WebDataModuleFromConfig(**config["data"]["params"])
        # from pudb import set_trace; set_trace()
        dataloader = datamod.train_dataloader()

        for i, batch in enumerate(dataloader):
            print(batch.keys())
            print(
                batch["image"].shape,
                f"max: {batch['image'].max()}, min: {batch['image'].min()}",
            )
            print(
                batch["img_feature"].shape,
                batch["img_feature"].max(),
                batch["img_feature"].min(),
            )
            print(f"Batch number: {i}")
            break
        print("end")

    elif False:
        from omegaconf import OmegaConf

        config = OmegaConf.load("datasets/debug_celebav.yaml")
        datamod = WebDataModuleFromConfig(**config["data"]["params"])
        # from pudb import set_trace; set_trace()
        dataloader = datamod.train_dataloader()

        def show_key_details(batch, key):
            print(key, batch[key].shape, batch[key].max(), batch[key].min())

        for i, batch in enumerate(dataloader):
            print(batch.keys())
            print(
                len(batch["frame_feature256"]),
                batch["frame_feature256"][0].shape,
                batch["frame_feature256"][0].max(),
                batch["frame_feature256"][0].min(),
            )
            show_key_details(batch, "emotions_caption_feature")
            print(batch["emotions_caption"])

            break
        print("end")
    elif True:
        from omegaconf import OmegaConf

        config = OmegaConf.load("datasets/debug_ucf101.yaml")
        datamod = WebDataModuleFromConfig(**config["data"]["params"])
        dataloader = datamod.train_dataloader()

        def show_key_details(batch, key):
            print(key, batch[key].shape, batch[key].max(), batch[key].min())

        for i, batch in enumerate(dataloader):
            print(batch.keys())
            print(
                len(batch["frame_feature256"]),
                batch["frame_feature256"][0].shape,
                batch["frame_feature256"][0].max(),
                batch["frame_feature256"][0].min(),
            )
            show_key_details(batch, "cls_id")
            show_key_details(batch, "cls_name")
            print(batch["emotions_caption"])
            break
        print("end")
    elif False:
        from omegaconf import OmegaConf

        config = OmegaConf.load("datasets/debug_laion_aesthetics6.yaml")
        datamod = WebDataModuleFromConfig(**config["data"]["params"])
        dataloader = datamod.train_dataloader()

        def show_key_details(batch, key):
            print(key, batch[key].shape, batch[key].max(), batch[key].min())

        for i, batch in enumerate(dataloader):
            print(batch.keys())
            show_key_details(batch, "jpg")
            print(batch["json"])
            print(batch["txt"])
            break
        print("end")
    elif False:
        from omegaconf import OmegaConf

        config = OmegaConf.load("datasets/debug_coco14.yaml")
        datamod = WebDataModuleFromConfig(**config["data"]["params"])
        dataloader = datamod.train_dataloader()

        def show_key_details(batch, key):
            print(key, batch[key].shape, batch[key].max(), batch[key].min())

        for i, batch in enumerate(dataloader):
            print(batch.keys())
            show_key_details(batch, "img_feature")
            show_key_details(batch, "caption_feature")
            show_key_details(batch, "image")

            break
        print("end")
