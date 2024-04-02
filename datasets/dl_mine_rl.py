# Source: plai-group/flexible-video-diffusion-modeling/datasets/minerl.py
# This code was copied from plai-group's flexible-video-diffusion-modeling, specifically the datasets/minerl.py file.
# For more details and license information, please refer to the original repository:
# Repository URL: https://github.com/plai-group/flexible-video-diffusion-modeling

import tensorflow as tf
import tensorflow_datasets as tfds
import minerl_navigate
import os
import numpy as np
from pathlib import Path

if __name__ == "__main__":
    data_dir = Path(".")  ### Changed ###
    orig_dataset = "minerl_navigate"
    torch_dataset_path = data_dir / f"{orig_dataset}-torch"
    torch_dataset_path.mkdir(exist_ok=True)

    for split in ["train", "test"]:
        torch_split_path = torch_dataset_path / split
        torch_split_path.mkdir(exist_ok=True)

        ds = tfds.load("minerl_navigate", data_dir=str(data_dir), shuffle_files=False)[
            split
        ]
        for cnt, item in enumerate(ds):
            video = item["video"].numpy()
            np.save(torch_split_path / f"{cnt}.npy", video)

        print(f" [-] {cnt} scenes in the {split} dataset")
