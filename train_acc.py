# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
"""
import random
import shutil
from einops import rearrange
from omegaconf import OmegaConf
from datasets.wds_dataloader import WebDataModuleFromConfig
import torch
from diffusers import StableDiffusionPipeline
from my_metrics import MyMetric

from utils.train_utils import (
    create_logger,
    get_latest_checkpoint,
    get_model,
    grad_clip,
    requires_grad,
    update_ema,
    wandb_runid_from_checkpoint,
)

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from copy import deepcopy
from time import time
import logging
import os
from tqdm import tqdm
import wandb
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
from utils.train_utils_args import rankzero_logging_info
import hydra
from hydra.core.hydra_config import HydraConfig
import accelerate
from wandb_utils import array2grid_pixel, get_max_ckpt_from_dir


def out2img(samples):
    return torch.clamp(127.5 * samples + 128.0, 0, 255).to(
        dtype=torch.uint8, device="cuda"
    )


def update_note(args, accelerator, slurm_job_id):
    args.note = (
        "v5"
        + str(args.note)
        + f"_{args.data.name}"
        + f"_{args.model.name}"
        + f"_bs{args.data.batch_size}"
        + f"_wd{args.optim.wd}"
        + f"_{accelerator.state.num_processes}g"
        + f"_{slurm_job_id}"
    )

    return args.note


def has_text(args):
    if "celebamm" in args.data.name:
        return True
    elif "coco" in args.data.name:
        return True
    else:
        return False


def is_video(args):
    if hasattr(args.model.params, "video_frames"):
        if args.model.params.video_frames > 0:
            return True
        elif args.model.params.video_frames == 0:
            return False
        else:
            raise ValueError("video_frames must be >= 0")
    else:
        return False


def init_zs(args, device, in_channels, input_size):
    if is_video(args):
        _zs = torch.randn(
            args.data.sample_fid_bs,
            args.model.params.video_frames,
            in_channels,
            input_size,
            input_size,
            device=device,
        )
    else:
        _zs = torch.randn(
            args.data.sample_fid_bs,
            in_channels,
            input_size,
            input_size,
            device=device,
        )
    return _zs


#################################################################################
#                                  Training Loop                                #
#################################################################################


@hydra.main(config_path="config", config_name="default", version_base=None)
def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    from accelerate.utils import AutocastKwargs

    if True:
        kwargs = AutocastKwargs(enabled=False)
        # https://github.com/pytorch/pytorch/issues/40497#issuecomment-709846922
        # https://github.com/huggingface/accelerate/issues/2487#issuecomment-1969997224
    else:
        kwargs = {}
    accelerator = accelerate.Accelerator(
        kwargs_handlers=[kwargs], mixed_precision=args.mixed_precision
    )
    device = accelerator.device
    accelerate.utils.set_seed(args.global_seed, device_specific=True)
    rank = accelerator.state.process_index
    logging.info(
        f"Starting rank={rank}, world_size={accelerator.state.num_processes}, device={device}."
    )
    is_multiprocess = True if accelerator.state.num_processes > 1 else False
    if accelerator.state.num_processes >= 4 * 8:
        args.data.sample_fid_n = min(args.data.sample_fid_n, 1_000)
        print(
            "decreasing sample_fid_n to 1_000 in node with >= 4*8 GPUs, an unknown bug from torchmetrics"
        )

    _fid_eval_batch_nums = args.data.sample_fid_n // (
        args.data.sample_fid_bs * accelerator.state.num_processes
    )
    assert _fid_eval_batch_nums > 0, f"{_fid_eval_batch_nums} <= 0"

    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    logging.info(f"slurm_job_id: {slurm_job_id}")

    local_bs = args.data.batch_size

    train_steps = 0

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logging.info(args)
        experiment_dir = HydraConfig.get().run.dir
        logging.info(f"Experiment directory created at {experiment_dir}")
        checkpoint_dir = (
            f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(rank, experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        if args.use_wandb:
            config_dict = OmegaConf.to_container(args, resolve=True)
            config_dict = {
                **config_dict,
                "experiment_dir": experiment_dir,
                "world_size": accelerator.state.num_processes,
                "local_batch_size": args.data.batch_size
                * accelerator.state.num_processes,
                "job_id": slurm_job_id,
            }
            extra_wb_kwargs = dict()
            if args.ckpt is not None:
                runid = wandb_runid_from_checkpoint(args.ckpt)
                extra_wb_kwargs["resume"] = "must"
                extra_wb_kwargs["id"] = runid
            args.note = update_note(
                args=args, accelerator=accelerator, slurm_job_id=slurm_job_id
            )
            wandb.init(
                project=args.wandb.project,
                name=args.note,
                config=config_dict,
                dir=experiment_dir,
                mode="online",
                **extra_wb_kwargs,
            )
            wandb_project_url = (
                f"https://wandb.ai/dpose-team/{wandb.run.project}/runs/{wandb.run.id}"
            )
            wandb_sync_command = (
                f"wandb sync {experiment_dir}/wandb/latest-run --append"
            )
    else:
        logger = create_logger(rank)

    best_fid = 666

    model, in_channels, input_size = get_model(args, device)

    try:
        assert (
            args.data.sample_fid_bs <= args.data.batch_size
        ), f"sample_fid_bs={args.data.sample_fid_bs} must be less than batch_size={args.data.batch_size}"
    except:
        args.data.sample_fid_bs = args.data.batch_size
        print(
            f"forced sample_fid_bs to be equal to batch_size={args.data.sample_fid_bs}"
        )

    ema_model = deepcopy(model).to(device)

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.optim.lr, weight_decay=args.optim.wd
    )

    update_ema(
        ema_model, model, decay=0
    )  # Ensure EMA is initialized with synced weights

    transport = create_transport(
        path_type=args.train.path_type,
        prediction=args.train.prediction,
        loss_weight=args.train.loss_weight,
        train_eps=args.train.train_eps,
        sample_eps=args.train.sample_eps,
    )  # default: velocity;
    transport_sampler = Sampler(transport)
    if args.is_latent:
        if has_text(args):
            image_model_id = "runwayml/stable-diffusion-v1-5"
            pipe = StableDiffusionPipeline.from_pretrained(
                image_model_id, local_files_only=False
            )
            vae = pipe.vae.to("cuda")
            vae.eval()
        else:
            vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(
                device
            )
            vae.eval()
    _param_amount = sum(p.numel() for p in model.parameters())

    logger.info(f"#parameters: {sum(p.numel() for p in model.parameters()):,}")

    datamod = WebDataModuleFromConfig(**args.data)
    loader = datamod.train_dataloader()

    loader, opt, model, ema_model = accelerator.prepare(loader, opt, model, ema_model)

    print("dtype:", model.final_layer.linear.weight.dtype)
    if args.ckpt is not None:
        args.ckpt = get_max_ckpt_from_dir(args.ckpt)

    if args.ckpt is not None:  # before accelerate.wrap()
        ckpt_path = args.ckpt
        state_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict["model"])
        model = model.to(device)
        ema_model.load_state_dict(state_dict["ema"])
        ema_model = ema_model.to(device)
        opt.load_state_dict(state_dict["opt"])

        logging.info("overriding args with checkpoint args")
        logging.info(args)
        train_steps = state_dict["train_steps"]
        best_fid = state_dict["best_fid"]
        logging.info(f"Loaded checkpoint from {ckpt_path}, train_steps={train_steps}")
        requires_grad(ema_model, False)
        if rank == 0:
            shutil.copy(ckpt_path, checkpoint_dir)

    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema_model.eval()  # EMA model should always be in eval mode

    log_steps = 0
    running_loss = 0
    start_time = time()

    sample_vis_n = args.data.sample_vis_n

    zs = torch.randn(sample_vis_n, in_channels, input_size, input_size, device=device)
    rankzero_logging_info(rank, f"zs shape: {zs.shape}")

    model_fn = ema_model.forward

    def get_data_generator():
        _init = train_steps
        while True:
            for data in tqdm(
                loader,
                disable=not accelerator.is_main_process,
                initial=_init,
                desc="train_steps",
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
                    elif "ucf101" in str(args.data.name):
                        yield data["frame_feature256"].to(device), data["cls_id"].to(
                            device
                        )
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

    def get_real_img_generator():  # [0,255]
        while True:
            for data in tqdm(
                loader,
                disable=not accelerator.is_main_process,
                initial=0,
                desc="generate_real_img",
            ):
                if has_text(args):
                    yield out2img(data["image"]).to(device)
                elif "facehq" in str(args.data.name):
                    yield out2img(data["image"]).to(device)
                elif "ucf101" in str(args.data.name):
                    _video = data["frame_feature256"].to(device)
                    if args.is_latent:
                        with torch.no_grad():
                            _video = torch.stack(
                                [
                                    vae.decode(_video[:, _v]).sample
                                    for _v in range(_video.shape[1])
                                ],
                                dim=1,
                            )
                    yield out2img(_video)
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
                    if args.is_latent:
                        with torch.no_grad():
                            _video = torch.stack(
                                [
                                    vae.decode(_video[:, _v]).sample
                                    for _v in range(_video.shape[1])
                                ],
                                dim=1,
                            )
                    yield out2img(_video)

                else:
                    raise NotImplementedError("latent data not supported")

    def get_cap_generator():
        while True:
            for data in tqdm(
                loader,
                disable=not accelerator.is_main_process,
                initial=0,
                desc="generate_captions",
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

    train_dg = get_data_generator()
    real_img_dg = get_real_img_generator()
    cap_dg = get_cap_generator()

    my_metric_kwargs = dict()
    if is_video(args):
        my_metric_kwargs = dict(
            choices=["fid", "fvd"], video_frame=args.model.params.video_frames
        )
    else:
        my_metric_kwargs = dict(choices=["fid", "is", "kid", "prdc", "sfid", "fdd"])
    my_metric = MyMetric(**my_metric_kwargs)

    if True:
        gt_recovered = next(real_img_dg)
        gt_recovered = accelerator.gather(gt_recovered.contiguous())
        if accelerator.is_main_process and args.use_wandb:
            if is_video(args):
                wandb_dict = {
                    "vis/gt_recovered": wandb.Video(
                        gt_recovered[:100].detach().cpu().numpy(), fps=1
                    )
                }
            else:
                wandb_dict = {
                    "vis/gt_recovered": wandb.Image(
                        array2grid_pixel(gt_recovered[:100])
                    )
                }
            wandb.log(wandb_dict)
            logging.info(wandb_project_url + "\n" + wandb_sync_command)

    while train_steps < args.data.train_steps:
        x, y = next(train_dg)
        x_img = x
        if args.is_latent:
            if args.use_latent:
                x = x.mul_(0.18215)
            else:
                with torch.no_grad():
                    x = vae.encode(x).latent_dist.sample().mul_(0.18215)

        model_kwargs = dict(y=y)
        loss_dict = transport.training_losses(model, x, model_kwargs)
        loss = loss_dict["loss"].mean()
        opt.zero_grad()
        accelerator.backward(loss)
        opt.step()
        grad_clip(opt, model, max_grad_norm=args.max_grad_norm)  # clip gradient
        update_ema(ema_model, model)

        running_loss += loss.item()
        log_steps += 1
        train_steps += 1
        if train_steps % args.log_every == 0:
            # Measure training speed:
            torch.cuda.synchronize()
            end_time = time()
            steps_per_sec = log_steps / (end_time - start_time)
            # Reduce loss history over all processes:
            avg_loss = torch.tensor(running_loss / log_steps, device=device)
            if is_multiprocess:
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss.item() / accelerator.state.num_processes
            if accelerator.is_main_process:
                logging.info(
                    f"(step={train_steps:07d}/{args.data.train_steps}), Best_FID: {best_fid}, Train Loss: {avg_loss:.4f}, BS-1GPU: {len(x_img)} Train Steps/Sec: {steps_per_sec:.2f}, slurm_job_id: {slurm_job_id} , {experiment_dir}"
                )
                logging.info(wandb_sync_command)
                latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
                logging.info(latest_checkpoint)
                logging.info(wandb_project_url)

                if args.use_wandb:
                    wandb_dict = {
                        "train_loss": avg_loss,
                        "train_steps_per_sec": steps_per_sec,
                        "best_fid": best_fid,
                        "bs_1gpu": len(x_img),
                        "param_amount": _param_amount,
                    }
                    wandb.log(
                        wandb_dict,
                        step=train_steps,
                    )
            # Reset monitoring variables:
            running_loss = 0
            log_steps = 0
            start_time = time()

        if train_steps % args.ckpt_every == 0 and train_steps > 0:
            if accelerator.is_main_process:
                checkpoint = {
                    "model": model.state_dict(),
                    "ema": ema_model.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args,
                    "train_steps": train_steps,
                    "best_fid": best_fid,
                }
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logging.info(f"Saved checkpoint to {checkpoint_path}")
            accelerator.wait_for_everyone()

        if train_steps % args.data.sample_fid_every == 0 and train_steps > 0:
            with torch.no_grad():  # very important
                torch.cuda.empty_cache()
                my_metric.reset()
                ##########
                logging.info("Generating real_img_4_fid...")
                if is_video(args):
                    for _ in range(args.data.sample_fid_n // local_bs):
                        _d = next(real_img_dg)
                        # for _ in range(_d.shape[1]):
                        _d = rearrange(_d, "b t c h w -> (b t) c h w")
                        my_metric.update_real(_d)

                else:
                    [
                        my_metric.update_real(next(real_img_dg))
                        for _ in range(args.data.sample_fid_n // local_bs)
                    ]
                real_img_4_fid = next(real_img_dg)
                print(f"real_img_4_fid: {my_metric._fid.real_features_num_samples}")
                ########
                logger.info(
                    f"Generating EMA samples, batch size_gpu = {args.data.sample_fid_bs}..."
                )
                sample_fn = transport_sampler.sample_ode()  # default to ode sampling

                vis_wandb_sample = None
                for _b_id in tqdm(
                    range(_fid_eval_batch_nums),
                    desc="sampling FID on the fly",
                    total=_fid_eval_batch_nums,
                ):
                    _zs = init_zs(args, device, in_channels, input_size)
                    if has_text(args):
                        ys, sam_caps = next(cap_dg)
                        ys, sam_caps = ys[: len(_zs)], sam_caps[: len(_zs)]
                        assert len(ys) == len(_zs), f"{len(ys)} != {len(_zs)}"
                        sample_model_kwargs = dict(y=ys.to(device))
                    elif "ucf101" in str(args.data.name):
                        ys = next(train_dg)[1][: len(_zs)]
                        sample_model_kwargs = dict(y=ys.to(device))
                    else:
                        sample_model_kwargs = dict()
                    try:
                        samples = sample_fn(_zs, model_fn, **sample_model_kwargs)[-1]
                    except Exception as e:
                        logging.info("sample_fn error", exc_info=True)
                        samples = torch.rand_like(_zs)

                    accelerator.wait_for_everyone()

                    if args.is_latent:
                        with torch.no_grad():
                            if is_video(args):
                                samples = torch.stack(
                                    [
                                        vae.decode(samples[_] / 0.18215).sample
                                        for _ in range(samples.shape[0])
                                    ],
                                    dim=1,
                                )
                            else:
                                samples = vae.decode(samples / 0.18215).sample

                            samples = out2img(samples)

                    out_sample_global = accelerator.gather(samples.contiguous())
                    if _b_id == 0:
                        vis_wandb_sample = out_sample_global
                        if has_text(args):
                            vis_sample_t2i_cap = sam_caps
                            vis_sample_t2i_img = samples.contiguous()
                    if is_video(args):
                        my_metric.update_fake(
                            rearrange(
                                out_sample_global,
                                "b t c h w -> (b t) c h w",
                            )
                        )
                    else:
                        my_metric.update_fake(out_sample_global)
                    del out_sample_global, samples, _zs
                    torch.cuda.empty_cache()

                ###
                _metric_dict = my_metric.compute()
                fid = _metric_dict["fid"]
                _metric_dict = {f"eval/{k}": v for k, v in _metric_dict.items()}
                best_fid = min(fid, best_fid)

                if accelerator.is_main_process:
                    logger.info(f"FID: {fid}, best_fid: {best_fid}")
                    if args.use_wandb:
                        wandb_dict = {
                            # "best_fid": best_fid,
                            f"best_fid{my_metric._fid.fake_features_num_samples}": best_fid,
                            "real_num": my_metric._fid.real_features_num_samples,
                            "fake_num": my_metric._fid.fake_features_num_samples,
                        }
                        wandb_dict.update(_metric_dict)

                        if is_video(args):
                            wandb_dict.update(
                                {
                                    "vis/gt": wandb.Video(
                                        real_img_4_fid[:100].detach().cpu().numpy(),
                                        fps=1,
                                    ),
                                    "vis/sample": wandb.Video(
                                        vis_wandb_sample[:100].detach().cpu().numpy(),
                                        fps=1,
                                    ),
                                }
                            )
                        else:
                            wandb_dict.update(
                                {
                                    "vis/gt": wandb.Image(
                                        array2grid_pixel(real_img_4_fid[:100])
                                    ),
                                    "vis/sample": wandb.Image(
                                        array2grid_pixel(vis_wandb_sample[:100])
                                    ),
                                }
                            )

                        if has_text(args):
                            wandb_dict["vis/sample_captions"] = [
                                wandb.Image(
                                    vis_sample_t2i_img[_], caption=vis_sample_t2i_cap[_]
                                )
                                for _ in range(len(vis_sample_t2i_cap))
                            ]

                        wandb.log(
                            wandb_dict,
                            step=train_steps,
                        )
                rankzero_logging_info(rank, "Generating EMA samples done.")
                torch.cuda.empty_cache()

    model.eval()

    logger.info("Done!")


if __name__ == "__main__":
    main()
