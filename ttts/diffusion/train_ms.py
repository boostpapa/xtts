from omegaconf import OmegaConf
import torchaudio
from ttts.diffusion.aa_model import AA_diffusion, denormalize_tacotron_mel, normalize_tacotron_mel
from ttts.gpt.voice_tokenizer import VoiceBpeTokenizer
from ttts.utils.diffusion import SpacedDiffusion, space_timesteps, get_named_beta_schedule
import torch
import copy
from datetime import datetime
import json
from vocos import Vocos
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from ttts.utils.infer_utils import load_model
from ttts.utils.utils import EMA, clean_checkpoints, plot_spectrogram_to_numpy, summarize, update_moving_average
from ttts.diffusion.dataset import DiffusionDataset, DiffusionCollater
from ttts.diffusion.model import DiffusionTts
import torch
import os
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import AdamW
from accelerate import Accelerator, DistributedDataParallelKwargs
import functools
import random

import torch
from torch.cuda.amp import autocast

from ttts.utils.diffusion import get_named_beta_schedule
from ttts.utils.resample import create_named_schedule_sampler, LossAwareSampler, DeterministicSampler, LossSecondMomentResampler
from ttts.utils.diffusion import space_timesteps, SpacedDiffusion
# from ttts.diffusion.diffusion_util import Diffuser
# from accelerate import DistributedDataParallelKwargs
from ttts.utils.utils import AttrDict, get_logger
from ttts.vqvae.xtts_dvae import DiscreteVAE
from ttts.gpt.model import UnifiedVoice
import argparse
import logging

logging.getLogger("numba").setLevel(logging.WARNING)


def do_spectrogram_diffusion(diffusion_model, diffuser, latents, conditioning_latents, temperature=1, verbose=True):
    """
    Uses the specified diffusion model to convert discrete codes into a spectrogram.
    """
    with torch.no_grad():
        output_seq_len = latents.shape[2] * 4 # This diffusion model converts from 22kHz spectrogram codes to a 24kHz spectrogram signal.
        output_shape = (latents.shape[0], 100, output_seq_len)

        noise = torch.randn(output_shape, device=latents.device) * temperature
        mel = diffuser.p_sample_loop(diffusion_model, output_shape, noise=noise,
                                    model_kwargs={
                                        "hint": latents,
                                        "refer": conditioning_latents
                                    },
                                    progress=verbose)
        return denormalize_tacotron_mel(mel)[:, :, :output_seq_len]


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def get_grad_norm(model):
    total_norm = 0
    for name, p in model.named_parameters():
        try:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        except:
            print(name)
            pass
    total_norm = total_norm ** (1. / 2) 
    return total_norm


def warmup(step):
    if step < 1000:
        return float(step/1000)
    else:
        return 1


class Trainer(object):
    def __init__(self, args):
        # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        self.cfg = OmegaConf.load(args.config)
        # self.cfg = json.load(open(cfg_path))
        trained_diffusion_steps = 1000
        self.trained_diffusion_steps = 1000
        desired_diffusion_steps = 1000
        self.desired_diffusion_steps = 1000
        cond_free_k = 2.

        self.train_dataset = DiffusionDataset(self.cfg, self.cfg.dataset['training_files'])
        self.eval_dataset = DiffusionDataset(self.cfg, self.cfg.dataset['validation_files'])
        self.train_dataloader = DataLoader(self.train_dataset, **self.cfg.dataloader,
                                           collate_fn=DiffusionCollater())
        self.eval_dataloader = DataLoader(self.eval_dataset, **self.cfg.dataloader,
                                          collate_fn=DiffusionCollater())

        self.train_steps = self.cfg['train']['train_steps']
        self.eval_interval = self.cfg.train['eval_interval']
        self.log_interval = self.cfg['train']['log_interval']
        self.num_epochs = self.cfg['train']['epochs']
        self.accum_grad = self.cfg['train']['accum_grad']
        self.use_fp16 = self.cfg['train']['fp16_run']
        precision = "fp16" if self.use_fp16 else "no" # ['no', 'fp8', 'fp16', 'bf16']

        self.diffusion = AA_diffusion(self.cfg)
        self.diffuser = SpacedDiffusion(
            use_timesteps=space_timesteps(trained_diffusion_steps, [desired_diffusion_steps]),
            model_mean_type='epsilon',
            model_var_type='learned_range', loss_type='mse',
            betas=get_named_beta_schedule('linear', trained_diffusion_steps),
            conditioning_free=False, conditioning_free_k=cond_free_k)
        # self.diffusion = DiffusionTts(**self.cfg['diffusion'])

        if 'pretrain_model' in self.cfg:
            model_path = self.cfg['pretrain_model']
            logging.warning("loading pretrain model: {}".format(model_path))
            diffusion_checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
            diffusion_checkpoint = diffusion_checkpoint['model'] if 'model' in diffusion_checkpoint else diffusion_checkpoint
            self.diffusion.load_state_dict(diffusion_checkpoint, strict=False)
            print(">> diffusion weights restored from:", model_path)

        ## load gpt model ##
        self.gpt = UnifiedVoice(**self.cfg.gpt)
        gpt_path = self.cfg.gpt_checkpoint
        gpt_checkpoint = torch.load(gpt_path, map_location=torch.device("cpu"))
        gpt_checkpoint = gpt_checkpoint['model'] if 'model' in gpt_checkpoint else gpt_checkpoint
        self.gpt.load_state_dict(gpt_checkpoint, strict=False)
        self.gpt.eval()
        print(">> GPT weights restored from:", gpt_path)
        self.mel_length_compression = self.gpt.mel_length_compression

        ## load gpt model ##
        self.dvae = DiscreteVAE(**self.cfg.vqvae)
        dvae_path = self.cfg.dvae_checkpoint
        dvae_checkpoint = torch.load(dvae_path, map_location=torch.device("cpu"))
        if 'model' in dvae_checkpoint:
            dvae_checkpoint = dvae_checkpoint['model']
        self.dvae.load_state_dict(dvae_checkpoint, strict=False)
        self.dvae.eval()
        print(">> vqvae weights restored from:", dvae_path)

        # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        self.accelerator = Accelerator(mixed_precision=precision, split_batches=True)
        self.model_dir = Path(args.model)
        if self.accelerator.is_main_process:
            self.model_dir.mkdir(exist_ok=True, parents=True)
        self.logger = get_logger(self.model_dir)

        self.optimizer = AdamW(self.diffusion.parameters(), lr=self.cfg['train']['lr'], betas=(0.9, 0.999), weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=warmup)
        self.diffusion, self.train_dataloader, self.eval_dataloader, self.optimizer, self.scheduler, self.gpt, self.dvae \
            = self.accelerator.prepare(self.diffusion, self.train_dataloader, self.eval_dataloader, self.optimizer, self.scheduler, self.gpt, self.dvae)
        self.grad_clip = self.cfg['train']['grad_clip']
        if self.grad_clip <= 0:
            self.grad_clip = 50
        self.global_step = 0

    def _get_target_encoder(self, model):
        target_encoder = copy.deepcopy(model)
        set_requires_grad(target_encoder, False)
        for p in target_encoder.parameters():
            p.DO_NOT_TRAIN = True
        return target_encoder

    def save_checkpoint(self, path):
        if self.accelerator.is_main_process:
            data = {
                'step': self.global_step,
                'model': self.accelerator.get_state_dict(self.diffusion),
            }
            torch.save(data, path)
            # unwrapped_model = self.accelerator.unwrap_model(self.gpt)
            # self.accelerator.save(unwrapped_model.state_dict(), path)
            # torch.save(unwrapped_model, path)

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return
        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.diffusion),
        }
        torch.save(data, str(self.logs_folder / f'model-{milestone}.pt'))

    def load(self, model_path):
        accelerator = self.accelerator
        device = accelerator.device
        data = torch.load(model_path, map_location=device)
        state_dict = data['model']
        self.step = data['step']
        model = self.accelerator.unwrap_model(self.diffusion)
        model.load_state_dict(state_dict)

    def eval(self):
        model = self.accelerator.unwrap_model(self.diffusion)
        device = self.accelerator.device
        model.eval()
        total_losses = 0.
        num_samples = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(self.eval_dataloader):
                input_data = [data['padded_mel_refer'], data['padded_text'], data['text_lengths'],
                                data['padded_mel'], data['wav_lens']]
                input_data[3] = self.dvae.get_codebook_indices(input_data[3])
                latent = self.gpt(*input_data, return_latent=True, clip_inputs=False).transpose(1, 2)
                # mel_recon_padded, mel_padded, mel_lengths, refer_padded, refer_lengths
                x_start = normalize_tacotron_mel(data['padded_mel'].to(device))
                aligned_conditioning = latent
                conditioning_latent = normalize_tacotron_mel(data['padded_mel_refer'].to(device))
                t = torch.randint(0, self.desired_diffusion_steps, (x_start.shape[0],), device=device).long().to(device)
                loss = self.diffuser.training_losses(
                    model=model,
                    x_start=x_start,
                    t=t,
                    model_kwargs={
                        "hint": aligned_conditioning,
                        "refer": conditioning_latent
                    },
                )["loss"].mean()
                num_sample = input_data[3].shape[0]
                num_samples += num_sample
                total_losses += loss * num_sample

        model.train()
        #total_losses /= num_samples
        return [total_losses]

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        if isinstance(self.dvae, torch.nn.parallel.DistributedDataParallel):
            self.dvae = self.dvae.module
            self.gpt = self.gpt.module

        if accelerator.is_main_process:
            writer = SummaryWriter(log_dir=self.model_dir)
            num_params = sum(p.numel() for p in self.dvae.parameters())
            print('the number of vqvae model parameters: {:,d}'.format(num_params))

            num_params = sum(p.numel() for p in self.gpt.parameters())
            print('the number of gpt model parameters: {:,d}'.format(num_params))

            num_params = sum(p.numel() for p in self.diffusion.parameters())
            print('the number of diffusion model parameters: {:,d}'.format(num_params))

            self.logger.info("Initial Evaluating ...")
            losses = self.eval()
            lr = self.optimizer.param_groups[0]["lr"]
            self.logger.info([x.item() for x in losses] + [self.global_step, lr])
            self.save_checkpoint(self.model_dir.joinpath(f"init.pth"))

        for epoch in range(0, self.num_epochs):
            total_losses = 0.
            for batch_idx, data in enumerate(self.train_dataloader):
                if data is None:
                    continue

                with torch.no_grad():
                    input_data = [data['padded_mel_refer'], data['padded_text'], data['text_lengths'],
                                    data['padded_mel'], data['wav_lens']]
                    input_data[3] = self.dvae.get_codebook_indices(input_data[3])
                    latent = self.gpt(*input_data, return_latent=True, clip_inputs=False).transpose(1, 2)

                # mel_recon_padded, mel_padded, mel_lengths, refer_padded, refer_lengths
                x_start = normalize_tacotron_mel(data['padded_mel'].to(device))
                aligned_conditioning = latent
                conditioning_latent = normalize_tacotron_mel(data['padded_mel_refer'].to(device))
                t = torch.randint(0, self.desired_diffusion_steps, (x_start.shape[0],), device=device).long().to(device)
                with self.accelerator.autocast():
                    loss = self.diffuser.training_losses(
                        model=self.diffusion,
                        x_start=x_start,
                        t=t,
                        model_kwargs={
                            "hint": aligned_conditioning,
                            "refer": conditioning_latent
                        },
                    )["loss"].mean()
                    unused_params =[]
                    model = self.accelerator.unwrap_model(self.diffusion)
                    unused_params.extend(list(model.refer_model.blocks.parameters()))
                    unused_params.extend(list(model.refer_model.out.parameters()))
                    unused_params.extend(list(model.refer_model.hint_converter.parameters()))
                    unused_params.extend(list(model.refer_enc.visual.proj))
                    extraneous_addition = 0
                    for p in unused_params:
                        extraneous_addition = extraneous_addition + p.mean()
                    loss = loss + 0*extraneous_addition
                    loss = loss / self.accum_grad
                accelerator.backward(loss)
                total_losses += loss

                if batch_idx % self.accum_grad != 0:
                    continue

                grad_norm = get_grad_norm(self.diffusion)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(self.diffusion.parameters(), self.grad_clip)
                accelerator.wait_for_everyone()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
                accelerator.wait_for_everyone()

                if self.global_step % self.log_interval == 0:
                    #logging.warning(f"batch size: {input_data[3].shape}")
                    lr = self.optimizer.param_groups[0]["lr"]
                    losses = [total_losses]
                    self.logger.info("Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(self.train_dataloader)
                    ))
                    self.logger.info([x.item() for x in losses] + [self.global_step, lr])

                if accelerator.is_main_process and self.global_step % self.eval_interval == 0:
                    self.logger.info("Evaluating ...")
                    losses = self.eval()
                    self.logger.info([x.item() for x in losses])
                    # self.save_checkpoint(self.model_dir.joinpath(f"model_{self.global_step}.pth"))
                    scalar_dict = {"loss": total_losses,
                                   "loss/grad": grad_norm}
                    summarize(
                        writer=writer,
                        global_step=self.global_step,
                        scalars=scalar_dict
                    )
                total_losses = 0.
                self.global_step += 1
            # one epoch training finish
            if accelerator.is_main_process:
                self.logger.info(f"Evaluating Epoch: {epoch}")
                losses = self.eval()
                lr = self.optimizer.param_groups[0]["lr"]
                self.logger.info([x.item() for x in losses] + [self.global_step, lr])
            self.save_checkpoint(self.model_dir.joinpath(f"epoch_{epoch}.pth"))
        accelerator.print('training complete')


def get_args():
    parser = argparse.ArgumentParser(description='train gpt')
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default='./configs/config.yaml',
        help='config file',
        required=True,
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="experiment base directory",
        default='exp',
        required=True,
    )
    # args = parser.parse_args()
    args, _ = parser.parse_known_args()

    return args


if __name__ == '__main__':
    args = get_args()
    print(args)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    trainer = Trainer(args)
    #trainer.load('/home/hyc/tortoise_plus_zh/ttts/diffusion/logs/2024-01-09-17-44-36/model-855.pt')

    trainer.train()
