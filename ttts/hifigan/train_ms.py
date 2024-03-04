import copy
from datetime import datetime
from inspect import signature
import json
from omegaconf import OmegaConf
from pathlib import Path
from accelerate import Accelerator
from tqdm import tqdm
from ttts.diffusion.diffusion_util import cycle, get_grad_norm, normalize_tacotron_mel
from ttts.diffusion.train import set_requires_grad
from ttts.hifigan.dataset import HiFiGANCollater, HifiGANDataset
from torch.utils.tensorboard import SummaryWriter
from ttts.hifigan.hifigan_discriminator import HifiganDiscriminator
from ttts.hifigan.hifigan_vocoder import HifiDecoder
from ttts.hifigan.losses import DiscriminatorLoss, GeneratorLoss
from ttts.utils.infer_utils import load_model
from ttts.utils.utils import EMA, clean_checkpoints, plot_spectrogram_to_numpy, summarize
import torch
from typing import Any, Callable, Dict, Union, Tuple
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import os

from ttts.vocoder.feature_extractors import MelSpectrogramFeatures
from ttts.utils.utils import AttrDict, get_logger
from ttts.utils.utils import make_pad_mask
from ttts.vqvae.xtts_dvae import DiscreteVAE
from ttts.gpt.model import UnifiedVoice
import argparse
import logging

logging.getLogger("numba").setLevel(logging.WARNING)


def warmup(step):
    if step < 1000:
        return float(step/1000)
    else:
        return 1


class Trainer(object):
    def __init__(self, args):
        # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        #json_config = json.load(open(args.config))
        #self.cfg = AttrDict(json_config)
        self.cfg = OmegaConf.load(args.config)

        self.train_dataset = HifiGANDataset(self.cfg, self.cfg.dataset['training_files'])
        self.eval_dataset = HifiGANDataset(self.cfg, self.cfg.dataset['validation_files'])
        self.train_dataloader = DataLoader(self.train_dataset, **self.cfg.dataloader,
                                           collate_fn=HiFiGANCollater())

        self.eval_dataloader = DataLoader(self.eval_dataset, **self.cfg.dataloader_eval,
                                          collate_fn=HiFiGANCollater())
        self.train_steps = self.cfg.train['train_steps']
        self.eval_interval = self.cfg.train['eval_interval']
        self.log_interval = self.cfg['train']['log_interval']
        self.num_epochs = self.cfg['train']['epochs']
        self.accum_grad = self.cfg['train']['accum_grad']
        self.lr = self.cfg['train']['lr']
        self.use_fp16 = self.cfg['train']['fp16_run']
        precision = "fp16" if self.use_fp16 else "no"  # ['no', 'fp8', 'fp16', 'bf16']

        self.hifigan_decoder = HifiDecoder(**self.cfg['hifigan'])
        self.hifigan_discriminator = HifiganDiscriminator()

        ## load gpt model ##
        self.gpt = UnifiedVoice(**self.cfg.gpt)
        gpt_path = self.cfg.gpt_checkpoint
        gpt_checkpoint = torch.load(gpt_path, map_location=torch.device("cpu"))
        gpt_checkpoint = gpt_checkpoint['model'] if 'model' in gpt_checkpoint else gpt_checkpoint
        self.gpt.load_state_dict(gpt_checkpoint, strict=False)
        self.gpt.eval()
        print(">> GPT weights restored from:", gpt_path)
        self.mel_length_compression = self.gpt.mel_length_compression

        ## load vqvae model ##
        self.dvae = DiscreteVAE(**self.cfg.vqvae)
        dvae_path = self.cfg.dvae_checkpoint
        dvae_checkpoint = torch.load(dvae_path, map_location=torch.device("cpu"))
        if 'model' in dvae_checkpoint:
            dvae_checkpoint = dvae_checkpoint['model']
        self.dvae.load_state_dict(dvae_checkpoint, strict=False)
        self.dvae.eval()
        print(">> vqvae weights restored from:", dvae_path)

        self.accelerator = Accelerator(mixed_precision=precision, split_batches=True)
        self.model_dir = Path(args.model)
        if self.accelerator.is_main_process:
            self.model_dir.mkdir(exist_ok=True, parents=True)
        self.logger = get_logger(self.model_dir)

        self.G_optimizer = AdamW(self.hifigan_decoder.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.01)
        self.D_optimizer = AdamW(self.hifigan_discriminator.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.01)
        self.G_scheduler = torch.optim.lr_scheduler.LambdaLR(self.G_optimizer, lr_lambda=warmup)
        self.D_scheduler = torch.optim.lr_scheduler.LambdaLR(self.D_optimizer, lr_lambda=warmup)
        self.hifigan_decoder, self.hifigan_discriminator, self.hifigan_decoder.speaker_encoder, self.train_dataloader, self.eval_dataloader, self.G_optimizer, self.D_optimizer, self.G_scheduler, self.D_scheduler, self.gpt, self.dvae \
            = self.accelerator.prepare(self.hifigan_decoder, self.hifigan_discriminator, self.hifigan_decoder.speaker_encoder, self.train_dataloader, self.eval_dataloader, self.G_optimizer, self.D_optimizer, self.G_scheduler, self.D_scheduler, self.gpt, self.dvae)
        self.mel_extractor = MelSpectrogramFeatures(**self.cfg.dataset['mel']).to(self.accelerator.device)
        self.disc_loss = DiscriminatorLoss()
        self.gen_loss = GeneratorLoss()
        self.grad_clip = self.cfg['train']['grad_clip']
        if self.grad_clip <= 0:
            self.grad_clip = 50
        self.global_step = 0

    def get_speaker_embedding(self, hifigan_decoder, audio, sr):
        audio_16k = torchaudio.functional.resample(audio, sr, 16000)
        return (
            hifigan_decoder.speaker_encoder.forward(audio_16k.to(self.accelerator.device), l2_norm=True)
            .unsqueeze(-1)
            .to(self.accelerator.device)
        )

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
                'model': self.accelerator.get_state_dict(self.hifigan_decoder),
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
            'model': self.accelerator.get_state_dict(self.hifigan_decoder),
        }
        torch.save(data, str(self.logs_folder / f'model-{milestone}.pt'))

    def load(self, model_path):
        accelerator = self.accelerator
        device = accelerator.device
        data = torch.load(model_path, map_location=device)
        state_dict = data['model']
        self.step = data['step']
        model = self.accelerator.unwrap_model(self.hifigan_decoder)
        model.load_state_dict(state_dict)

    def eval(self):
        hifigan_decoder = self.accelerator.unwrap_model(self.hifigan_decoder)
        hifigan_discriminator = self.accelerator.unwrap_model(self.hifigan_discriminator)
        device = self.accelerator.device
        hifigan_decoder.eval()
        hifigan_discriminator.eval()
        d_losses = 0.
        g_losses = 0.
        num_samples = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(self.eval_dataloader):
                for key in data:
                    data[key] = data[key].to(device)

                padded_mel_code = self.dvae.get_codebook_indices(data['padded_mel'])
                latent = self.gpt(data['padded_mel_refer'], data['padded_text'],
                                  data['text_lengths'], padded_mel_code,
                                  data['wav_lens'],
                                  return_latent=True, clip_inputs=False)

                mel_codes_lens = torch.ceil(data['wav_lens'] / self.mel_length_compression).long()
                mask_pad = make_pad_mask(mel_codes_lens).unsqueeze(2)
                latent = latent.masked_fill_(mask_pad, 0.0)
                latent = latent.transpose(1, 2)

                x = latent
                y = data['padded_wav']

                # discriminator loss
                g = self.get_speaker_embedding(hifigan_decoder, data['padded_wav_refer'], 24000)
                y_hat = hifigan_decoder(x, g)
                score_fake, feat_fake = hifigan_discriminator(y_hat.detach())
                score_real, feat_real = hifigan_discriminator(y.clone())
                loss_d = self.disc_loss(score_fake, score_real)['loss']

                # generator loss
                score_fake, feat_fake = hifigan_discriminator(y_hat)
                loss_g = self.gen_loss(y_hat, y, score_fake, feat_fake, feat_real)['loss']

                num_sample = y.shape[0]
                num_samples += num_sample
                d_losses += loss_d * num_sample
                g_losses += loss_g * num_sample

        hifigan_decoder.train()
        hifigan_discriminator.train()
        d_losses /= num_samples
        g_losses /= num_samples
        total_losses = d_losses + g_losses
        return [total_losses, d_losses, g_losses]

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        if isinstance(self.dvae, torch.nn.parallel.DistributedDataParallel):
            self.dvae = self.dvae.module
            self.gpt = self.gpt.module

        if accelerator.is_main_process:
            self.logger.info(self.cfg)
            writer = SummaryWriter(log_dir=self.model_dir)
            num_params = sum(p.numel() for p in self.dvae.parameters())
            print('the number of vqvae model parameters: {:,d}'.format(num_params))

            num_params = sum(p.numel() for p in self.gpt.parameters())
            print('the number of gpt model parameters: {:,d}'.format(num_params))

            num_params = sum(p.numel() for p in self.hifigan_decoder.parameters())
            print('the number of hifigan_decoder model parameters: {:,d}'.format(num_params))

            self.logger.info("Initial Evaluating ...")
            losses = self.eval()
            lr = self.G_optimizer.param_groups[0]["lr"]
            self.logger.info([x.item() for x in losses] + [self.global_step, lr])
            self.save_checkpoint(self.model_dir.joinpath(f"init.pth"))

        for epoch in range(0, self.num_epochs):
            total_losses = 0.
            for batch_idx, data in enumerate(self.train_dataloader):
                # 'padded_text': padded_text,
                # 'padded_mel': padded_mel,
                # 'padded_wav': padded_wav,
                # 'padded_mel_refer': padded_mel_refer,
                # 'padded_wav_refer': padded_wav_refer,
                if data is None:
                    continue

                with torch.no_grad():
                    for key in data:
                        data[key] = data[key].to(device)

                    padded_mel_code = self.dvae.get_codebook_indices(data['padded_mel'])
                    latent = self.gpt(data['padded_mel_refer'], data['padded_text'],
                                      data['text_lengths'], padded_mel_code,
                                      data['wav_lens'],
                                      return_latent=True, clip_inputs=False)

                mel_codes_lens = torch.ceil(data['wav_lens'] / self.mel_length_compression).long()
                mask_pad = make_pad_mask(mel_codes_lens).unsqueeze(2)
                latent = latent.masked_fill_(mask_pad, 0.0)
                latent = latent.transpose(1, 2)

                x = latent
                y = data['padded_wav']
                with self.accelerator.autocast():
                    g = self.get_speaker_embedding(self.hifigan_decoder, data['padded_wav_refer'], 24000)
                    y_hat = self.hifigan_decoder(x, g)
                    score_fake, feat_fake = self.hifigan_discriminator(y_hat.detach())
                    score_real, feat_real = self.hifigan_discriminator(y.clone())
                    loss_d = self.disc_loss(score_fake, score_real)['loss']

                total_losses += loss_d
                self.accelerator.backward(loss_d)
                grad_norm_d = get_grad_norm(self.hifigan_discriminator)
                accelerator.clip_grad_norm_(self.hifigan_discriminator.parameters(), self.grad_clip)
                accelerator.wait_for_everyone()
                self.D_optimizer.step()
                self.D_optimizer.zero_grad()
                self.D_scheduler.step()
                accelerator.wait_for_everyone()

                score_fake, feat_fake = self.hifigan_discriminator(y_hat)
                loss_g = self.gen_loss(y_hat, y, score_fake, feat_fake, feat_real)['loss']
                total_losses += loss_g
                self.accelerator.backward(loss_g)
                grad_norm_g = get_grad_norm(self.hifigan_decoder)
                accelerator.clip_grad_norm_(self.hifigan_decoder.parameters(), self.grad_clip)
                accelerator.wait_for_everyone()
                self.G_optimizer.step()
                self.G_optimizer.zero_grad()
                self.G_scheduler.step()
                accelerator.wait_for_everyone()

                if self.global_step % self.log_interval == 0:
                    #logging.warning(f"batch size: {input_data[3].shape}")
                    lr = self.G_optimizer.param_groups[0]["lr"]
                    losses = [total_losses, loss_d, loss_g]
                    self.logger.info("Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(self.train_dataloader)
                    ))
                    self.logger.info([x.item() for x in losses] + [self.global_step, lr])

                if accelerator.is_main_process and self.global_step % self.eval_interval == 0:
                    self.logger.info("Evaluating ...")
                    losses = self.eval()
                    self.logger.info([x.item() for x in losses])
                    # self.save_checkpoint(self.model_dir.joinpath(f"model_{self.global_step}.pth"))
                    scalar_dict = {"loss_d": loss_d, "loss/grad_d": grad_norm_d,
                                   "loss_g": loss_d, "loss/grad_g": grad_norm_g,}
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
                lr = self.G_optimizer.param_groups[0]["lr"]
                self.logger.info([x.item() for x in losses] + [self.global_step, lr])
            self.save_checkpoint(self.model_dir.joinpath(f"epoch_{epoch}.pth"))
        accelerator.print('training complete')


def get_args():
    parser = argparse.ArgumentParser(description='train gpt')
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default='./configs/config.json',
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

    trainer.train()
