from omegaconf import OmegaConf
import json
import os
import time
import random
import argparse
import itertools
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from ttts.utils.utils import AttrDict, get_logger
from ttts.utils.lr_scheduler import CosineLRScheduler

from setproctitle import setproctitle
from ttts.utils.checkpoint import load_checkpoint as load_checkpoint_xtts
from ttts.vocoder.feature_extractors import MelSpectrogramFeatures
from dataset import BigVGANDataset, BigVGANCollator
from bigvgan import BigVGAN
from msf.msf_disc import Discriminators as MSFDiscriminator
from discriminators import (
    MultiPeriodDiscriminator,
    MultiResolutionDiscriminator,
    MultiBandDiscriminator,
    MultiScaleSubbandCQTDiscriminator,
)
from loss import (
    feature_loss,
    generator_loss,
    discriminator_loss,
    MultiScaleMelSpectrogramLoss,
)

from utils import (
    plot_spectrogram,
    plot_spectrogram_clipped,
    scan_checkpoint,
    load_checkpoint,
    save_checkpoint,
    save_audio,
)

import auraloss
import logging

from ttts.gpt.model import UnifiedVoice
from ttts.vqvae.xtts_dvae import DiscreteVAE
logging.getLogger("numba").setLevel(logging.WARNING)

torch.backends.cudnn.benchmark = False


class Trainer(object):
    def __init__(self, args):
        # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        if args.config.endswith(".json"):
            json_config = json.load(open(args.config))
            self.cfg = AttrDict(json_config)
        else:
            self.cfg = OmegaConf.load(args.config)

        self.train_dataset = BigVGANDataset(self.cfg, self.cfg.dataset['training_files'])
        self.eval_dataset = BigVGANDataset(self.cfg, self.cfg.dataset['validation_files'], is_eval=True)
        self.train_dataloader = DataLoader(self.train_dataset, **self.cfg.dataloader,
                                           collate_fn=BigVGANCollator())
        self.eval_dataloader = DataLoader(self.eval_dataset, **self.cfg.dataloader,
                                          collate_fn=BigVGANCollator())

        self.train_steps = self.cfg.train['train_steps']
        self.eval_interval = self.cfg.train['eval_interval']
        self.save_interval = self.cfg.train['save_interval'] if 'save_interval' in self.cfg.train else None
        self.log_interval = self.cfg.train['log_interval']
        self.num_epochs = self.cfg.train['epochs']
        self.accum_grad = self.cfg.train['accum_grad']
        self.freeze_step = self.cfg.train['freeze_step'] if 'freeze_step' in self.cfg.train else 0
        self.lr = self.cfg.train['lr']
        self.precision = self.cfg.train['precision']
        # ['no', 'fp8', 'fp16', 'bf16']
        precision = self.precision
        if self.precision == "fp32":
            precision = "no"
        print(">> training precision:", precision)

        self.accelerator = Accelerator(mixed_precision=precision, split_batches=True)
        self.model_dir = Path(args.model)
        if self.accelerator.is_main_process:
            self.model_dir.mkdir(exist_ok=True, parents=True)
            print(f"Checkpoints directory: {self.model_dir}")
        self.logger = get_logger(self.model_dir)

        self.global_step = 0
        self.start_epoch = 0

        h = self.cfg.bigvgan
        # Define BigVGAN generator
        self.generator = BigVGAN(h)

        # Define discriminators. MPD is used by default
        self.mpd = MultiPeriodDiscriminator(self.cfg.bigvgan)

        # Define additional discriminators. BigVGAN-v1 uses UnivNet's MRD as default
        # New in BigVGAN-v2: option to switch to new discriminators: MultiBandDiscriminator / MultiScaleSubbandCQTDiscriminator
        if h.get("use_mbd_instead_of_mrd", False):  # Switch to MBD
            print(
                "[INFO] using MultiBandDiscriminator of BigVGAN-v2 instead of MultiResolutionDiscriminator"
            )
            # Variable name is kept as "mrd" for backward compatibility & minimal code change
            self.mrd = MultiBandDiscriminator(h)
        elif h.get("use_cqtd_instead_of_mrd", False):  # Switch to CQTD
            print(
                "[INFO] using MultiScaleSubbandCQTDiscriminator of BigVGAN-v2 instead of MultiResolutionDiscriminator"
            )
            self.mrd = MultiScaleSubbandCQTDiscriminator(h)
        else:  # Fallback to original MRD in BigVGAN-v1
            self.mrd = MultiResolutionDiscriminator(h)

        self.msfd = MSFDiscriminator(stacks=4, channels=64, kernel_size=9,
                                     frequency_ranges=[[0, 40], [20, 60], [40, 80], [60, 100]])

        # New in BigVGAN-v2: option to switch to multi-scale L1 mel loss
        if h.get("use_multiscale_melloss", False):
            print(
                "[INFO] using multi-scale Mel l1 loss of BigVGAN-v2 instead of the original single-scale loss"
            )
            self.fn_mel_loss_multiscale = MultiScaleMelSpectrogramLoss(
                sampling_rate=h.sampling_rate
            )  # NOTE: accepts waveform as input
        else:
            self.fn_mel_loss_singlescale = F.l1_loss

        self.lambda_melloss = h.get(
            "lambda_melloss", 45.0
        )  # Defaults to 45 in BigVGAN-v1 if not set

        if h.mel_type == "pytorch":
            self.mel_pytorch = MelSpectrogramFeatures(sample_rate=h.sampling_rate,
                                                     n_fft=h.n_fft,
                                                     hop_length=h.hop_size,
                                                     win_length=h.win_size,
                                                     n_mels=h.num_mels,
                                                     mel_fmin=h.fmin, )
            print(f"Warning use torchaudio.transforms.MelSpectrogram extract mel.")

        if os.path.isdir(self.model_dir):
            # New in v2.1: If the step prefix pattern-based checkpoints are not found, also check for renamed files in Hugging Face Hub to resume training
            cp_g = scan_checkpoint(self.model_dir, prefix="g_", renamed_file="bigvgan_generator.pt")
            cp_do = scan_checkpoint(self.model_dir, prefix="do_", renamed_file="bigvgan_discriminator_optimizer.pt",)

            # Load the latest checkpoint if exists
            self.steps = 0
            if cp_g is None or cp_do is None:
                state_dict_do = None
            else:
                state_dict_g = load_checkpoint(cp_g, "cpu")
                state_dict_do = load_checkpoint(cp_do, "cpu")
                self.generator.load_state_dict(state_dict_g["generator"])
                self.mpd.load_state_dict(state_dict_do["mpd"])
                self.mrd.load_state_dict(state_dict_do["mrd"])
                self.msfd.load_state_dict(state_dict_do['msfd'])
                self.global_step = state_dict_do["steps"] + 1
                self.start_epoch = state_dict_do["epoch"]

        ## load gpt model ##
        self.gpt = UnifiedVoice(**self.cfg.gpt)
        gpt_path = self.cfg.gpt_checkpoint
        gpt_checkpoint = torch.load(gpt_path, map_location=torch.device("cpu"))
        gpt_checkpoint = gpt_checkpoint['model'] if 'model' in gpt_checkpoint else gpt_checkpoint
        self.gpt.load_state_dict(gpt_checkpoint, strict=True)
        self.gpt.eval()
        print(">> GPT weights restored from:", gpt_path)
        self.mel_length_compression = self.gpt.mel_length_compression

        ## load vqvae model ##
        self.dvae = DiscreteVAE(**self.cfg.vqvae)
        dvae_path = self.cfg.dvae_checkpoint
        dvae_checkpoint = torch.load(dvae_path, map_location=torch.device("cpu"))
        if 'model' in dvae_checkpoint:
            dvae_checkpoint = dvae_checkpoint['model']
        self.dvae.load_state_dict(dvae_checkpoint, strict=True)
        self.dvae.eval()
        print(">> vqvae weights restored from:", dvae_path)

        self.optim_g = torch.optim.AdamW(
            self.generator.parameters(), self.lr, betas=[h.adam_b1, h.adam_b2]
        )
        self.optim_d = torch.optim.AdamW(
            itertools.chain(self.mrd.parameters(), self.mpd.parameters(), self.msfd.parameters()),
            self.lr,
            betas=[h.adam_b1, h.adam_b2],
        )

        if state_dict_do is not None:
            self.optim_g.load_state_dict(state_dict_do["optim_g"])
            self.optim_d.load_state_dict(state_dict_do["optim_d"])

        global total_training_steps
        total_batches = len(self.train_dataloader)
        total_training_steps = total_batches*self.num_epochs/self.accum_grad
        print(f">> total training epoch: {self.num_epochs}, batches per epoch: {total_batches}, steps: {total_training_steps}")
        global final_lr_ratio
        global num_warmup_step
        if 'min_lr' in self.cfg.train:
            self.min_lr = self.cfg.train['min_lr']
            num_warmup_step = self.cfg.train['warmup_steps']
            final_lr_ratio = self.min_lr / self.lr

        self.scheduler_g = CosineLRScheduler(self.optim_g, warmup_steps=num_warmup_step,
                                             total_steps=total_training_steps, lr_min_ratio=final_lr_ratio)
        self.scheduler_g.set_step(self.global_step)
        self.scheduler_d = CosineLRScheduler(self.optim_d, warmup_steps=num_warmup_step,
                                             total_steps=total_training_steps, lr_min_ratio=final_lr_ratio)
        self.scheduler_d.set_step(self.global_step)

        self.generator, self.mpd, self.mrd, self.msfd, \
            self.train_dataloader, self.eval_dataloader, \
            self.optim_g, self.optim_d, self.scheduler_g, self.scheduler_d, \
            self.gpt, self.dvae, self.mel_pytorch \
            = self.accelerator.prepare(self.generator, self.mpd, self.mrd, self.msfd,
                                       self.train_dataloader, self.eval_dataloader,
                                       self.optim_g, self.optim_d, self.scheduler_g, self.scheduler_d,
                                       self.gpt, self.dvae, self.mel_pytorch)
        self.grad_clip = self.cfg.train['grad_clip']
        if self.grad_clip <= 0:
            self.grad_clip = 1000

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        setproctitle("test_bigvgan")
        if isinstance(self.dvae, torch.nn.parallel.DistributedDataParallel):
            self.dvae = self.dvae.module
            self.gpt = self.gpt.module

        if accelerator.is_main_process:
            self.logger.info(self.cfg)
            #writer = SummaryWriter(log_dir=self.model_dir)
            num_params = sum(p.numel() for p in self.dvae.parameters())
            print('the number of vqvae model parameters: {:,d}'.format(num_params))

            num_params = sum(p.numel() for p in self.gpt.parameters())
            print('the number of gpt model parameters: {:,d}'.format(num_params))

            print(self.generator)
            print(self.mpd)
            print(self.mrd)
            print(self.msfd)
            print(f"Generator params: {sum(p.numel() for p in self.generator.parameters())}")
            print(f"Discriminator mpd params: {sum(p.numel() for p in self.mpd.parameters())}")
            print(f"Discriminator mrd params: {sum(p.numel() for p in self.mrd.parameters())}")
            print(f"Discriminator msfd params: {sum(p.numel() for p in self.msfd.parameters())}")
            '''
            self.logger.info("Initial Evaluating ...")
            losses = self.eval()
            lr = self.optimizer.param_groups[0]["lr"]
            self.logger.info([x.item() for x in losses] + [self.global_step, lr])
            #self.save_checkpoint(self.model_dir.joinpath(f"init.pth"), lr, self.start_epoch, self.global_step)
            '''

        segment_size = self.cfg.bigvgan.segment_size  # 11264 # 8192 # 24576
        hop_length = self.cfg.bigvgan.gpt_dim
        chunk = segment_size // hop_length

        for epoch in range(self.start_epoch, self.num_epochs):
            for i, batch in enumerate(self.train_dataloader):
                start_b = time.time()
                for key in batch:
                    batch[key] = batch[key].to(device)

                text = batch['padded_text']
                text_lens = batch['text_lengths']
                mel_refer = batch['padded_mel_refer']
                mel_refer_len = batch['mel_refer_lens']
                mel_infer = batch['padded_mel_infer']
                mel_infer_len = batch['mel_infer_lens']
                wav_infer = batch['padded_wav_infer']
                wav_infer_lens = batch['wav_infer_lens']
                wav_refer = batch['padded_wav_refer']
                wav_refer_lens = batch['wav_refer_lens']

                y_ = wav_infer.squeeze(1)
                mel_ref = mel_refer

                with torch.no_grad():
                    mel_code = self.dvae.get_codebook_indices(mel_infer)
                    latent = self.gpt(mel_refer,
                                     text,
                                     text_lens,
                                     mel_code,
                                     wav_infer_lens,
                                     cond_mel_lengths=mel_refer_len,
                                     return_latent=True,
                                     clip_inputs=False,)
                    # latent = latent / std
                    latent = latent.transpose(1, 2)

                    x = []
                    y = []
                    # print(f"y_ shape {y_.shape}, wav_infer_lens {wav_infer_lens}")
                    for wav, feat, len_ in zip(y_, latent, wav_infer_lens):
                        # [T], [1024, T/1024], 1
                        start = 0
                        if len_ // 1024 - 1 > chunk:
                            start = random.randint(0, len_ // 1024 - 1 - chunk)
                        gpt_latent = feat[:, start:start + chunk]
                        # print(f"wav shape {wav.shape}")
                        wav = wav[start * hop_length: (start + chunk) * hop_length]
                        # print(f"gpt_latent shape {gpt_latent.shape}, wav shape {wav.shape}")

                        x.append(gpt_latent)
                        y.append(wav)

                    x = torch.stack(x)
                    y = torch.stack(y)

                    y_mel = self.mel_pytorch(y)
                    feats_lengths = torch.LongTensor([segment_size // 256 + 1] * y_mel.size(0))
                    y = y.unsqueeze(1)

                # Discriminators
                self.optim_d.zero_grad()

                with accelerator.autocast():
                    y_g_hat, contrastive_loss = self.generator(x.transpose(1, 2), mel_ref.transpose(1, 2), mel_refer_len)
                    y_g_hat_mel = self.mel_pytorch(y_g_hat.squeeze(1))

                    # MPD
                    y_df_hat_r, y_df_hat_g, _, _ = self.mpd(y, y_g_hat.detach())
                    loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(
                        y_df_hat_r, y_df_hat_g
                    )

                    # MRD
                    y_ds_hat_r, y_ds_hat_g, _, _ = self.mrd(y, y_g_hat.detach())
                    loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(
                        y_ds_hat_r, y_ds_hat_g
                    )

                    # MSFD
                    res_fake = self.msfd(y_g_hat_mel.detach().transpose(1, 2), feats_lengths)
                    res_real = self.msfd(y_mel.transpose(1, 2), feats_lengths)
                    real_loss = torch.stack([torch.mean((1 - w) ** 2) for w in res_real]).sum()
                    fake_loss = torch.stack([torch.mean(w ** 2) for w in res_fake]).sum()
                    loss_disc_msf = real_loss + fake_loss
                    loss_disc_all = loss_disc_s + loss_disc_f + loss_disc_msf

                # Whether to freeze D for initial training steps
                if self.global_step >= self.freeze_step:
                    #loss_disc_all.backward()
                    accelerator.backward(loss_disc_all)
                    if accelerator.sync_gradients:
                        grad_norm_mpd = accelerator.clip_grad_norm_(self.mpd.parameters(), self.grad_clip)
                        grad_norm_mrd = accelerator.clip_grad_norm_(self.mrd.parameters(), self.grad_clip)
                        grad_norm_msfd = accelerator.clip_grad_norm_(self.msfd.parameters(), self.grad_clip)
                    accelerator.wait_for_everyone()
                    self.optim_d.step()
                else:
                    print(f"[WARNING] skipping D training for the first {self.freeze_step} steps")
                    grad_norm_mpd = 0.0
                    grad_norm_mrd = 0.0
                    grad_norm_msfd = 0.

                # Generator
                self.optim_g.zero_grad()

                if self.cfg.bigvgan.get("use_multiscale_melloss", False):  # uses wav <y, y_g_hat> for loss
                    loss_mel = self.fn_mel_loss_multiscale(y, y_g_hat) * self.lambda_melloss
                else:  # Uses mel <y_mel, y_g_hat_mel> for loss
                    loss_mel = self.fn_mel_loss_singlescale(y_mel, y_g_hat_mel) * self.lambda_melloss

                with accelerator.autocast():
                    # MPD loss
                    y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(y, y_g_hat)
                    loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                    loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)

                    # MRD loss
                    y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.mrd(y, y_g_hat)
                    loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
                    loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)

                    # msfd loss
                    d_res = self.msfd(y_g_hat_mel.transpose(1, 2), feats_lengths)
                    loss_adv_msfd = torch.stack([torch.mean((1 - w) ** 2) for w in d_res]).sum()

                    if self.global_step >= self.freeze_step:
                        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel + loss_adv_msfd
                    else:
                        print(f"[WARNING] using regression loss only for G for the first {self.freeze_step} steps")
                        loss_gen_all = loss_mel

                #loss_gen_all.backward()
                accelerator.backward(loss_gen_all)
                if accelerator.sync_gradients:
                    grad_norm_g = accelerator.clip_grad_norm_(self.generator.parameters(), self.grad_clip)
                accelerator.wait_for_everyone()
                self.optim_g.step()

                if accelerator.is_main_process:
                    if self.global_step % self.log_interval == 0:
                        mel_error = loss_mel.item() / self.lambda_melloss
                        # Log training mel regression loss to stdout
                        self.logger.info(
                            f"Epoch: {epoch + 1:d}, "
                            f"Steps: {self.global_step:d}, "
                            f"Gen Loss Total: {loss_gen_all:4.3f}, "
                            f"Mel Error: {mel_error:4.3f}, "
                            f"s/b: {time.time() - start_b:4.3f} "
                            f"lr: {self.optim_g.param_groups[0]['lr']:4.7f} "
                            f"grad_norm_g: {grad_norm_g:4.3f}"
                        )
                    # Checkpointing
                    #if self.global_step % self.save_interval == 0:
                    if self.global_step % self.save_interval == 0 and self.global_step != 0:
                        checkpoint_path = f"{self.model_dir}/g_{self.global_step:08d}"
                        save_checkpoint(
                            checkpoint_path,
                            {
                                "generator": self.accelerator.get_state_dict(self.generator),
                            },
                        )
                        checkpoint_path = f"{self.model_dir}/do_{self.global_step:08d}"
                        save_checkpoint(
                            checkpoint_path,
                            {
                                "mpd": self.accelerator.get_state_dict(self.mpd),
                                "mrd": self.accelerator.get_state_dict(self.mrd),
                                'msfd': self.accelerator.get_state_dict(self.msfd),
                                "optim_g": self.accelerator.get_state_dict(self.optim_g),
                                "optim_d": self.accelerator.get_state_dict(self.optim_d),
                                "steps": self.global_step,
                                "epoch": epoch,
                            },
                        )
                # end main_process
                self.global_step += 1
                self.scheduler_g.step()
                self.scheduler_d.step()

        accelerator.print('training complete')
        accelerator.end_training()


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
