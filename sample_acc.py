# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained SiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import random
from einops import rearrange
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from datasets.wds_dataloader import WebDataModuleFromConfig
from my_metrics import MyMetric
from train_acc import has_text, is_video
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import hydra
from utils.train_utils import get_model, requires_grad
import accelerate
import wandb
from diffusers import StableDiffusionPipeline



@hydra.main(config_path="config", config_name="default", version_base=None)
def main(args):
    """
    Run sampling.
    """
    sample_mode = args.sample_mode
    torch.backends.cuda.matmul.allow_tf32 = (
        args.allow_tf32
    )  # True: fast but may lead to some small numerical differences
    assert (
        torch.cuda.is_available()
    ), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    from accelerate.utils import AutocastKwargs

    if True:
        kwargs = AutocastKwargs(enabled=False)
        # https://github.com/pytorch/pytorch/issues/40497#issuecomment-709846922
        # https://github.com/huggingface/accelerate/issues/2487#issuecomment-1969997224
    else:
        kwargs = {}
    accelerator = accelerate.Accelerator(kwargs_handlers=[kwargs])
    device = accelerator.device
    accelerate.utils.set_seed(args.global_seed, device_specific=True)
    rank = accelerator.state.process_index
    print(
        f"Starting rank={rank}, world_size={accelerator.state.num_processes}, device={device}."
    )
    

    assert args.ckpt is not None, "Must specify a checkpoint to sample from"
    model, in_channels, input_size = get_model(args, device)
    if rank == 0:
        print(f"in_channels={in_channels}, input_size={input_size}")

    if True:
        state_dict = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        _model_dict = state_dict["ema"]
        _model_dict = {k.replace("module.", ""): v for k, v in _model_dict.items()}
        model.load_state_dict(_model_dict)
        model = model.to(device)
        requires_grad(model, False)
        if rank == 0:
            print(f"Loaded checkpoint from {args.ckpt}")
    

    model.eval()  # important!
    if is_video(args):
        _metric = MyMetric(choices=["fvd"], device=device)
        print("using videos metrics")
    else:
        _metric = MyMetric(
            choices=["fid",],
            device=device,
        )
        print("using image metrics")

    local_bs = args.offline_sample_local_bs
    args.data.batch_size = local_bs  # used for generating captions,etc.
    args.data.num_workers = 1
    print("local_bs", local_bs)
    datamod = WebDataModuleFromConfig(**args.data)
    loader = datamod.train_dataloader()

    loader, model = accelerator.prepare(loader, model)

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    global_bs = local_bs * accelerator.state.num_processes
    total_samples = int(math.ceil(args.num_fid_samples / global_bs) * global_bs)
    assert (
        total_samples % accelerator.state.num_processes == 0
    ), "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // accelerator.state.num_processes)
    assert (
        samples_needed_this_gpu % local_bs == 0
    ), "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // local_bs)
    if rank == 0:
        print(
            f"Total number of images that will be sampled: {total_samples} with global_batch_size={global_bs}"
        )

    def get_cap_generator():
        while True:
            for data in tqdm(
                loader,
                disable=not (rank == 0),
                initial=0,
                desc=f"generate_captions, for iters {iterations}",
            ):
                if has_text(args):
                    _cap_feats, _cap = (
                        data["caption_feature"].to(device),
                        data["caption"],
                    )
                    B, N, T, C = _cap_feats.shape  # each image has N captions
                    _p = random.randint(0, N - 1)
                    yield _cap_feats[:, _p], [_cap[i][_p] for i in range(len(_cap))]
                else:
                    raise NotImplementedError("current dataset doesnt have captions")

    if has_text(args):
        cap_dg = get_cap_generator()

    transport = create_transport(
        args.train.path_type,
        args.train.prediction,
        args.train.loss_weight,
        args.train.train_eps,
        args.train.sample_eps,
    )
    sampler = Sampler(transport)
    if sample_mode == "ODE":
        if args.likelihood:
            assert (
                args.cfg_scale == 1
            ), f"Likelihood is incompatible with guidance, but cfg_scale={args.cfg_scale} was provided."
            sample_fn = sampler.sample_ode_likelihood(
                sampling_method=args.ode.sampling_method,
                num_steps=args.ode.num_sampling_steps,
                atol=args.ode.atol,
                rtol=args.ode.rtol,
            )
        else:
            sample_fn = sampler.sample_ode(
                sampling_method=args.ode.sampling_method,
                num_steps=args.ode.num_sampling_steps,
                atol=args.ode.atol,
                rtol=args.ode.rtol,
                reverse=args.ode.reverse,
            )
    elif sample_mode == "SDE":
        sample_fn = sampler.sample_sde(
            sampling_method=args.sampling_method,
            diffusion_form=args.diffusion_form,
            diffusion_norm=args.diffusion_norm,
            last_step=args.last_step,
            last_step_size=args.last_step_size,
            num_steps=args.num_sampling_steps,
        )
    else:
        raise ValueError(f"Unknown sample_mode: {sample_mode}")
    if args.is_latent:
        if has_text(args):
            image_model_id = "runwayml/stable-diffusion-v1-5"
            pipe = StableDiffusionPipeline.from_pretrained(
                image_model_id, local_files_only=False
            )
            vae = pipe.vae.to("cuda")
            vae.eval()
            print("Loaded VAE from RunwayML",image_model_id)
        else:
            vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(
                device
            )
            vae.eval()
            print(f"Loaded VAE from stabilityai/sd-vae-ft-{args.vae}")
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"

    # Create folder to save samples:
    ckpt_string_name = (
        os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    )
    if sample_mode == "ODE":
        folder_name = "_".join(
            [
                args.data.name,
                args.model.name,
                ckpt_string_name,
                "llh1" if args.likelihood else "llh0",
                f"bs{args.offline_sample_local_bs}",
                sample_mode,
                str(args.ode.num_sampling_steps),
                args.ode.sampling_method,
            ]
        )

    elif sample_mode == "SDE":
        folder_name = "_".join(
            [
                args.data.name,
                args.model.name,
                ckpt_string_name,
                f"bs{args.offline_sample_local_bs}",
                sample_mode,
                str(args.num_sampling_steps),
                args.sde.sampling_method,
                args.sde.diffusion_form,
                str(args.sde.last_step),
                str(args.sde.last_step_size),
            ]
        )
    else:
        raise ValueError(f"Unknown sample_mode: {sample_mode}")

    sample_folder_dir = f"{args.sample_dir}/{folder_name}"

    if rank == 0:
        if args.use_wandb:
            entity = args.wandb.entity
            project = args.wandb.project + "_vis"
            print(f"Logging to wandb entity={entity}, project={project},rank={rank}")
            config_dict = OmegaConf.to_container(args, resolve=True)
            wandb.init(
                project=project,
                name=folder_name,
                config=config_dict,
                dir=sample_folder_dir,
                resume="allow",
                mode="online",
            )
    
            wandb_project_url = (
                f"https://wandb.ai/dpose-team/{wandb.run.project}/runs/{wandb.run.id}"
            )
            wandb_sync_command = f"wandb sync {sample_folder_dir}/wandb/latest-run --append"
            wandb_desc = "\n".join(
                [
                    "*" * 24,
                    str(config_dict),
                    folder_name,
                    wandb_project_url,
                    wandb_sync_command,
                    "*" * 24,
                ]
            )
        else:
            wandb_project_url='wandb_project_url_null'
            wandb_sync_command='wandb_sync_command_null'
            wandb_desc='wandb_desc_null'
            
            
    

    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    accelerator.wait_for_everyone()

    pbar = range(iterations)
    pbar = tqdm(pbar, total=iterations, desc="sampling") if rank == 0 else pbar
    total = 0

    def get_data_generator():
        while True:
            for data in tqdm(
                loader,
                disable=not (rank == 0),
                initial=0,
                desc=f"generate_images, for iters {iterations}",
            ):
                if args.use_latent:
                    if has_text(args):
                        _cap_feats = data["caption_feature"]
                        B, N, T, C = _cap_feats.shape  # each image has N captions
                        yield data["img_feature"].to(device), _cap_feats[
                            :, random.randint(0, N - 1)
                        ].to(device)
                    elif "facehq" in str(args.data.name):
                        yield data["latent"].to(device), None
                    elif "church" in str(args.data.name):
                        yield data["latent"].to(device), None
                    elif "ucf101" in str(args.data.name):
                        yield data["frame_feature256"].to(device), data["cls_id"]
                    elif "celebav" in str(args.data.name):
                        _start = random.randint(
                            0,
                            data["frame_feature256"].shape[1]
                            - args.model.params.video_frames
                            - 1,
                        )
                        _video = data["frame_feature256"][
                            :, _start : _start + args.model.params.video_frames
                        ].to(device)
                        yield _video, None
                    else:
                        raise NotImplementedError(
                            f"latent data not supported, args.data.name={args.data.name}"
                        )
                else:
                    yield data["image"].to(device), None

    data_generator = get_data_generator()

    samples_pil = lambda samples: torch.clamp(127.5 * samples + 128.0, 0, 255).to(
        device="cuda"
    )

    if rank == 0:
        print(wandb_desc)

    for bs_index in pbar:
        if is_video(args):
            z = torch.randn(
                local_bs,
                args.data.video_frames,
                in_channels,
                input_size,
                input_size,
                device=device,
            )
        else:
            z = torch.randn(
                local_bs,
                in_channels,
                input_size,
                input_size,
                device=device,
            )
        if args.data.num_classes > 0:
            y = torch.randint(0, args.data.num_classes, (local_bs,), device=device)
        elif has_text(args):
            y, _caption_texts = next(cap_dg)
            y, _caption_texts = (
                y[:local_bs],
                _caption_texts[:local_bs],
            )
            assert len(y) == local_bs, f"{len(y)} != {local_bs}"
            y = y.to(device)
        else:
            y = None

        model_kwargs = dict(y=y)
        model_fn = model.forward
        gts, _y = next(data_generator)

        with torch.no_grad():
            samples = sample_fn(z, model_fn, **model_kwargs)[-1]
            if args.is_latent:
                if not is_video(args):
                    samples = vae.decode(samples / 0.18215).sample
                    gts = vae.decode(
                        gts
                    ).sample  # no need for / 0.18215, because gts is not from the input of flow matching, it from the directed sampled latens from LDM
                else:
                    samples = torch.stack(
                        [
                            vae.decode(samples[_] / 0.18215).sample
                            for _ in range(len(samples))
                        ],
                        dim=1,
                    )
                    gts = torch.stack(
                        [vae.decode(gts[_]).sample for _ in range(len(gts))], dim=1
                    )
        gts = gts[: len(samples)]

        sam_4fid, gts_4fid = samples_pil(samples), samples_pil(gts)

        _metric.update_real(gts_4fid.to(dtype=torch.uint8))
        _metric.update_fake(sam_4fid.to(dtype=torch.uint8))

        # Save samples to disk as individual .png files
        for _iii, sample in enumerate(sam_4fid):
            index = _iii * accelerator.state.num_processes + rank + total
            Image.fromarray(
                rearrange(sample, "c w h -> w h c").to("cpu", dtype=torch.uint8).numpy()
            ).save(f"{sample_folder_dir}/{index:06d}.png")

        if args.use_wandb and bs_index <= 5:
            if rank == 0:
                if is_video(args):
                    captions_sample = [str(_) for _ in _y]
                    wandb.log(
                        {
                            f"vis/samples_single": [
                                wandb.Video(sam_4fid[i], caption=captions_sample[i])
                                for i in range(len(sam_4fid))
                            ],
                            f"vis/gts_single": [
                                wandb.Video(gts_4fid[i], caption=captions_sample[i])
                                for i in range(len(gts_4fid))
                            ],
                        },
                        step=bs_index,
                    )
                    print("log_image into wandb")
                else:
                    if has_text(args):
                        # captions_gt = ["none" if _y is None else _y[i] for i in range(len(_y))]
                        captions_sample = _caption_texts
                    else:
                        captions_sample = [
                            f"null caption" for _ in range(len(sam_4fid))
                        ]
                    wandb.log(
                        {
                            f"vis/samples_single": [
                                wandb.Image(sam_4fid[i], caption=captions_sample[i])
                                for i in range(len(sam_4fid))
                            ],
                            f"vis/gts_single": [
                                wandb.Image(gts_4fid[i]) for i in range(len(gts_4fid))
                            ],
                        },
                        step=bs_index,
                    )
                    print("log_image into wandb")
            accelerator.wait_for_everyone()
            if True:
                sam_4fid = accelerator.gather(sam_4fid)
                gts_4fid = accelerator.gather(gts_4fid)
                if rank == 0:
                    wandb.log(
                        {
                            f"vis/samples": wandb.Image(sam_4fid),
                            f"vis/gts": wandb.Image(gts_4fid),
                        },
                        step=bs_index,
                    )
                    print("log_image into wandb")
            accelerator.wait_for_everyone()

        total += global_bs
        accelerator.wait_for_everyone()
        if bs_index >= 3 and args.sample_debug:
            print("sample_debug, break at bs_index", bs_index)
            break

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    accelerator.wait_for_everyone()
    _metric_result = _metric.compute()
    _fid = _metric_result["fid"]
    print(f"FID: {_fid}")
    print(_metric_result)
    _metric_result = {f"eval/{k}": v for k, v in _metric_result.items()}
    if rank == 0:
        wandb.log(_metric_result)

    accelerator.wait_for_everyone()


if __name__ == "__main__":

    main()
