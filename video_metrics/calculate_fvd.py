import numpy as np
import torch
from tqdm import tqdm


def trans(x):
    # if greyscale images add channel
    if x.shape[-3] == 1:
        x = x.repeat(1, 1, 3, 1, 1)

    # permute BTCHW -> BCTHW
    x = x.permute(0, 2, 1, 3, 4)

    return x


def calculate_fvd(videos1, videos2, device, method="styleganv"):

    if method == "styleganv":
        from fvd.styleganv.fvd import (
            get_fvd_feats,
            frechet_distance,
            load_i3d_pretrained,
        )
    elif method == "videogpt":
        from fvd.videogpt.fvd import load_i3d_pretrained
        from fvd.videogpt.fvd import get_fvd_logits as get_fvd_feats
        from fvd.videogpt.fvd import frechet_distance

    print("calculate_fvd...")

    # videos [batch_size, timestamps, channel, h, w]
    batch_size, timestamps, channel, h, w = videos1.shape
    assert (
        timestamps >= 10
    ), f"timestamps must >= 10 according to {https://github.com/JunyaoHu/common_metrics_on_video_quality}"

    assert videos1.shape == videos2.shape

    i3d = load_i3d_pretrained(device=device)

    # support grayscale input, if grayscale -> channel*3
    # BTCHW -> BCTHW
    # videos -> [batch_size, channel, timestamps, h, w]

    videos1 = trans(videos1)
    videos2 = trans(videos2)

    clip_timestamp = timestamps - 1  # only consider last timestamp

    # get a video clip
    # videos_clip [batch_size, channel, timestamps[:clip], h, w]
    videos_clip1 = videos1[:, :, :clip_timestamp]
    videos_clip2 = videos2[:, :, :clip_timestamp]

    # get FVD features
    feats1 = get_fvd_feats(videos_clip1, i3d=i3d, device=device)
    feats2 = get_fvd_feats(videos_clip2, i3d=i3d, device=device)

    # calculate FVD when timestamps[:clip]
    _fvd = frechet_distance(feats1, feats2)

    print("FVD: ", _fvd)
    return _fvd


# test code / using example


def main():
    NUMBER_OF_VIDEOS = 8
    VIDEO_LENGTH = 50
    CHANNEL = 3
    SIZE = 64
    videos1 = torch.zeros(
        NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False
    )
    videos2 = torch.ones(
        NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False
    )
    device = torch.device("cuda")
    # device = torch.device("cpu")

    import json

    result = calculate_fvd(videos1, videos2, device, method="videogpt")
    print(json.dumps(result, indent=4))

    result = calculate_fvd(videos1, videos2, device, method="styleganv")
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()
