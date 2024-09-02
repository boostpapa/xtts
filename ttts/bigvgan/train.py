# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.

# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.


import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
from omegaconf import OmegaConf
import itertools
import os
import random
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from env import AttrDict, build_env
from meldataset import mel_spectrogram, MAX_WAV_VALUE
from dataset import BigVGANDataset, BigVGANCollator
from ttts.vocoder.feature_extractors import MelSpectrogramFeatures

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
import torchaudio as ta
from pesq import pesq
from tqdm import tqdm
import auraloss

from ttts.gpt.model import UnifiedVoice
from ttts.vqvae.xtts_dvae import DiscreteVAE
from ttts.utils.checkpoint import load_checkpoint as load_checkpoint_xtts


torch.backends.cudnn.benchmark = False


def train(rank, a, h):
    if h.num_gpus > 1:
        # initialize distributed
        init_process_group(
            backend=h.dist_config["dist_backend"],
            init_method=h.dist_config["dist_url"],
            world_size=h.dist_config["world_size"] * h.num_gpus,
            rank=rank,
        )

    # Set seed and device
    torch.cuda.manual_seed(h.seed)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank:d}")

    # Define BigVGAN generator
    generator = BigVGAN(h).to(device)

    # Define discriminators. MPD is used by default
    mpd = MultiPeriodDiscriminator(h).to(device)

    # Define additional discriminators. BigVGAN-v1 uses UnivNet's MRD as default
    # New in BigVGAN-v2: option to switch to new discriminators: MultiBandDiscriminator / MultiScaleSubbandCQTDiscriminator
    if h.get("use_mbd_instead_of_mrd", False):  # Switch to MBD
        print(
            "[INFO] using MultiBandDiscriminator of BigVGAN-v2 instead of MultiResolutionDiscriminator"
        )
        # Variable name is kept as "mrd" for backward compatibility & minimal code change
        mrd = MultiBandDiscriminator(h).to(device)
    elif h.get("use_cqtd_instead_of_mrd", False):  # Switch to CQTD
        print(
            "[INFO] using MultiScaleSubbandCQTDiscriminator of BigVGAN-v2 instead of MultiResolutionDiscriminator"
        )
        mrd = MultiScaleSubbandCQTDiscriminator(h).to(device)
    else:  # Fallback to original MRD in BigVGAN-v1
        mrd = MultiResolutionDiscriminator(h).to(device)

    msfd = MSFDiscriminator(stacks=4, channels=64, kernel_size=9, frequency_ranges=[[0, 40], [20, 60], [40, 80], [60, 100]]).to(device)

    # New in BigVGAN-v2: option to switch to multi-scale L1 mel loss
    if h.get("use_multiscale_melloss", False):
        print(
            "[INFO] using multi-scale Mel l1 loss of BigVGAN-v2 instead of the original single-scale loss"
        )
        fn_mel_loss_multiscale = MultiScaleMelSpectrogramLoss(
            sampling_rate=h.sampling_rate
        )  # NOTE: accepts waveform as input
    else:
        fn_mel_loss_singlescale = F.l1_loss

    if h.mel_type == "pytorch":
        mel_pytorch = MelSpectrogramFeatures(sample_rate=h.sampling_rate,
                                             n_fft=h.n_fft,
                                             hop_length=h.hop_size,
                                             win_length=h.win_size,
                                             n_mels=h.num_mels,
                                             mel_fmin=h.fmin, ).to(device)
        print(f"Warning use torchaudio.transforms.MelSpectrogram extract mel.")

    # Print the model & number of parameters, and create or scan the latest checkpoint from checkpoints directory
    if rank == 0:
        print(generator)
        print(mpd)
        print(mrd)
        print(f"Generator params: {sum(p.numel() for p in generator.parameters())}")
        print(f"Discriminator mpd params: {sum(p.numel() for p in mpd.parameters())}")
        print(f"Discriminator mrd params: {sum(p.numel() for p in mrd.parameters())}")
        print(f"Discriminator msfd params: {sum(p.numel() for p in msfd.parameters())}")
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print(f"Checkpoints directory: {a.checkpoint_path}")

    if os.path.isdir(a.checkpoint_path):
        # New in v2.1: If the step prefix pattern-based checkpoints are not found, also check for renamed files in Hugging Face Hub to resume training
        cp_g = scan_checkpoint(
            a.checkpoint_path, prefix="g_", renamed_file="bigvgan_generator.pt"
        )
        cp_do = scan_checkpoint(
            a.checkpoint_path,
            prefix="do_",
            renamed_file="bigvgan_discriminator_optimizer.pt",
        )

    # Load the latest checkpoint if exists
    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g["generator"])
        mpd.load_state_dict(state_dict_do["mpd"])
        mrd.load_state_dict(state_dict_do["mrd"])
        msfd.load_state_dict(state_dict_do['msfd'])
        steps = state_dict_do["steps"] + 1
        last_epoch = state_dict_do["epoch"]

    # Initialize DDP, optimizers, and schedulers
    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        mrd = DistributedDataParallel(mrd, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(
        generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2]
    )
    optim_d = torch.optim.AdamW(
        itertools.chain(mrd.parameters(), mpd.parameters()),
        h.learning_rate,
        betas=[h.adam_b1, h.adam_b2],
    )

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do["optim_g"])
        optim_d.load_state_dict(state_dict_do["optim_d"])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=h.lr_decay, last_epoch=last_epoch
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=h.lr_decay, last_epoch=last_epoch
    )

    ## load vqvae model ##
    cfg = OmegaConf.load(h.gpt_config)

    dvae = DiscreteVAE(**cfg.vqvae)
    dvae_path = cfg.dvae_checkpoint
    load_checkpoint_xtts(dvae, dvae_path)
    dvae = dvae.to(device)
    dvae.eval()
    print(">> vqvae weights restored from:", dvae_path)

    ## load gpt model ##
    gpt = UnifiedVoice(**cfg.gpt)
    gpt_path = cfg.gpt_checkpoint
    load_checkpoint_xtts(gpt, gpt_path)
    gpt = gpt.to(device)
    gpt.eval()
    print(">> GPT weights restored from:", gpt_path)
    gpt.post_init_gpt2_config(use_deepspeed=False, kv_cache=False, half=False)

    trainset = BigVGANDataset(h.sampling_rate, h.training_files)
    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None
    train_loader = DataLoader(trainset,
                              batch_size=h.batch_size,
                              shuffle=False,
                              num_workers=h.num_workers,
                              drop_last=True,
                              pin_memory=True,
                              sampler=train_sampler,
                              collate_fn=BigVGANCollator())

    if rank == 0:
        validset = BigVGANDataset(h.sampling_rate, h.validation_files)
        validation_loader = DataLoader(validset,
                                       batch_size=1,
                                       shuffle=False,
                                       num_workers=1,
                                       drop_last=True,
                                       pin_memory=True,
                                       collate_fn=BigVGANCollator())

        # Tensorboard logger
        sw = SummaryWriter(os.path.join(a.checkpoint_path, "logs"))
        if a.save_audio:  # Also save audio to disk if --save_audio is set to True
            os.makedirs(os.path.join(a.checkpoint_path, "samples"), exist_ok=True)

    """
    Validation loop, "mode" parameter is automatically defined as (seen or unseen)_(name of the dataset).
    If the name of the dataset contains "nonspeech", it skips PESQ calculation to prevent errors 
    """

    def validate(rank, a, h, loader, mode="seen"):
        assert rank == 0, "validate should only run on rank=0"
        generator.eval()
        torch.cuda.empty_cache()

        val_err_tot = 0
        val_pesq_tot = 0
        val_mrstft_tot = 0

        # Modules for evaluation metrics
        pesq_resampler = ta.transforms.Resample(h.sampling_rate, 16000).cuda()
        loss_mrstft = auraloss.freq.MultiResolutionSTFTLoss(device="cuda")

        if a.save_audio:  # Also save audio to disk if --save_audio is set to True
            os.makedirs(
                os.path.join(a.checkpoint_path, "samples", f"gt_{mode}"),
                exist_ok=True,
            )
            os.makedirs(
                os.path.join(a.checkpoint_path, "samples", f"{mode}_{steps:08d}"),
                exist_ok=True,
            )

        with torch.no_grad():
            print(f"step {steps} {mode} speaker validation...")

            # Loop over validation set and compute metrics
            for j, batch in enumerate(tqdm(loader)):
                x, y, _, y_mel = batch
                y = y.to(device)
                if hasattr(generator, "module"):
                    y_g_hat = generator.module(x.to(device))
                else:
                    y_g_hat = generator(x.to(device))
                y_mel = y_mel.to(device, non_blocking=True)
                y_g_hat_mel = mel_spectrogram(
                    y_g_hat.squeeze(1),
                    h.n_fft,
                    h.num_mels,
                    h.sampling_rate,
                    h.hop_size,
                    h.win_size,
                    h.fmin,
                    h.fmax_for_loss,
                )
                min_t = min(y_mel.size(-1), y_g_hat_mel.size(-1))
                val_err_tot += F.l1_loss(y_mel[..., :min_t], y_g_hat_mel[..., :min_t]).item()

                # PESQ calculation. only evaluate PESQ if it's speech signal (nonspeech PESQ will error out)
                if (
                    not "nonspeech" in mode
                ):  # Skips if the name of dataset (in mode string) contains "nonspeech"

                    # Resample to 16000 for pesq
                    y_16k = pesq_resampler(y)
                    y_g_hat_16k = pesq_resampler(y_g_hat.squeeze(1))
                    y_int_16k = (y_16k[0] * MAX_WAV_VALUE).short().cpu().numpy()
                    y_g_hat_int_16k = (
                        (y_g_hat_16k[0] * MAX_WAV_VALUE).short().cpu().numpy()
                    )
                    val_pesq_tot += pesq(16000, y_int_16k, y_g_hat_int_16k, "wb")

                # MRSTFT calculation
                min_t = min(y.size(-1), y_g_hat.size(-1))
                val_mrstft_tot += loss_mrstft(y_g_hat[...,:min_t], y[...,:min_t]).item()

                # Log audio and figures to Tensorboard
                if j % a.eval_subsample == 0:  # Subsample every nth from validation set
                    if steps >= 0:
                        sw.add_audio('gt_{}/y_{}'.format(mode, j), y[0], steps, h.sampling_rate)
                        if a.save_audio:  # also save audio to disk if --save_audio is set to True
                            save_audio(y[0], os.path.join(a.checkpoint_path, 'samples', 'gt_{}'.format(mode),
                                                          '{:04d}.wav'.format(j)), h.sampling_rate)
                        sw.add_figure('gt_{}/y_spec_{}'.format(mode, j), plot_spectrogram(x[0]), steps)

                    sw.add_audio('generated_{}/y_hat_{}'.format(mode, j), y_g_hat[0], steps, h.sampling_rate)
                    if a.save_audio:  # also save audio to disk if --save_audio is set to True
                        save_audio(y_g_hat[0, 0],
                                   os.path.join(a.checkpoint_path, 'samples', '{}_{:08d}'.format(mode, steps),
                                                '{:04d}.wav'.format(j)), h.sampling_rate)
                    # spectrogram of synthesized audio
                    y_hat_spec = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                                                 h.sampling_rate, h.hop_size, h.win_size,
                                                 h.fmin, h.fmax)
                    sw.add_figure('generated_{}/y_hat_spec_{}'.format(mode, j),
                                  plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()), steps)
                    # visualization of spectrogram difference between GT and synthesized audio
                    # difference higher than 1 is clipped for better visualization
                    spec_delta = torch.clamp(torch.abs(x[0] - y_hat_spec.squeeze(0).cpu()), min=1e-6, max=1.)
                    sw.add_figure('delta_dclip1_{}/spec_{}'.format(mode, j),
                                  plot_spectrogram_clipped(spec_delta.numpy(), clip_max=1.), steps)

            val_err = val_err_tot / (j + 1)
            val_pesq = val_pesq_tot / (j + 1)
            val_mrstft = val_mrstft_tot / (j + 1)
            # Log evaluation metrics to Tensorboard
            sw.add_scalar(f"validation_{mode}/mel_spec_error", val_err, steps)
            sw.add_scalar(f"validation_{mode}/pesq", val_pesq, steps)
            sw.add_scalar(f"validation_{mode}/mrstft", val_mrstft, steps)

        generator.train()

    # Exit the script if --evaluate is set to True
    if a.evaluate:
        exit()

    segment_size = h.segment_size  # 11264 # 8192 # 24576
    hop_length = h.gpt_dim
    chunk = segment_size // hop_length

    # Main training loop
    generator.train()
    mpd.train()
    mrd.train()
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print(f"Epoch: {epoch + 1}")

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
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

            y_ = wav_infer
            if h.mel_type == "pytorch":
                mel_ref = mel_refer
            else:
                mel_ref = mel_spectrogram(wav_refer.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size,
                                          h.win_size, h.fmin, h.fmax_for_loss)

            with torch.no_grad():
                mel_code = dvae.get_codebook_indices(mel_infer)
                latent = gpt(mel_refer,
                             text,
                             text_lens,
                             mel_code,
                             wav_infer_lens,
                             cond_mel_lengths=mel_refer_len,
                             return_latent=True,
                             clip_inputs=False,)
                #latent = latent / std
                latent = latent.transpose(1, 2)

                x = []
                y = []
                for wav, feat, len_ in zip(y_, latent, wav_infer_lens):
                    # [T], [1024, T/1024], 1
                    start = 0
                    if len_ // 1024 - 1 > chunk:
                        start = random.randint(0, len_ // 1024 - 1 - chunk)
                    gpt_latent = feat[:, start:start + chunk]
                    wav = wav[start * hop_length: (start + chunk) * hop_length]

                    x.append(gpt_latent)
                    y.append(wav)

                x = torch.stack(x)
                y = torch.stack(y)

                if h.mel_type == "pytorch":
                    y_mel = mel_pytorch(y)
                else:
                    y_mel = mel_spectrogram(y, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin,
                                        h.fmax_for_loss)
                feats_lengths = torch.LongTensor([segment_size // 256 + 1] * y_mel.size(0))

            '''
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_mel = y_mel.to(device, non_blocking=True)
            mel_ref = mel_ref.to(device, non_blocking=True)
            '''

            y = y.unsqueeze(1)
            y_g_hat, contrastive_loss = generator(x.transpose(1, 2), mel_ref.transpose(1, 2), mel_refer_len)
            if h.mel_type == "pytorch":
                y_mel = mel_pytorch(y_g_hat.squeeze(1))
            else:
                y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size,
                                              h.win_size, h.fmin, h.fmax_for_loss)

            optim_d.zero_grad()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(
                y_df_hat_r, y_df_hat_g
            )

            # MRD
            y_ds_hat_r, y_ds_hat_g, _, _ = mrd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(
                y_ds_hat_r, y_ds_hat_g
            )

            # MSFD
            res_fake = msfd(y_g_hat_mel.detach().transpose(1, 2), feats_lengths)
            res_real = msfd(y_mel.transpose(1, 2), feats_lengths)
            real_loss = torch.stack([torch.mean((1 - w) ** 2) for w in res_real]).sum()
            fake_loss = torch.stack([torch.mean(w ** 2) for w in res_fake]).sum()
            loss_disc_msf = real_loss + fake_loss

            loss_disc_all = loss_disc_s + loss_disc_f + loss_disc_msf

            # Set clip_grad_norm value
            clip_grad_norm = h.get("clip_grad_norm", 1000.0)  # Default to 1000

            # Whether to freeze D for initial training steps
            if steps >= a.freeze_step:
                loss_disc_all.backward()
                grad_norm_mpd = torch.nn.utils.clip_grad_norm_(mpd.parameters(), 1000.)
                grad_norm_mrd = torch.nn.utils.clip_grad_norm_(mrd.parameters(), 1000.)
                grad_norm_msfd = torch.nn.utils.clip_grad_norm_(msfd.parameters(), 1000.)
                optim_d.step()
            else:
                print(
                    f"[WARNING] skipping D training for the first {a.freeze_step} steps"
                )
                grad_norm_mpd = 0.0
                grad_norm_mrd = 0.0
                grad_norm_msfd = 0.

            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss
            lambda_melloss = h.get(
                "lambda_melloss", 45.0
            )  # Defaults to 45 in BigVGAN-v1 if not set
            if h.get("use_multiscale_melloss", False):  # uses wav <y, y_g_hat> for loss
                loss_mel = fn_mel_loss_multiscale(y, y_g_hat) * lambda_melloss
            else:  # Uses mel <y_mel, y_g_hat_mel> for loss
                loss_mel = fn_mel_loss_singlescale(y_mel, y_g_hat_mel) * lambda_melloss

            # MPD loss
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)

            # MRD loss
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = mrd(y, y_g_hat)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)

            # msfd loss
            d_res = msfd(y_g_hat_mel.transpose(1,2), feats_lengths)
            loss_adv_msfd = torch.stack([torch.mean((1-w)**2) for w in d_res]).sum()

            if steps >= a.freeze_step:
                loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel + loss_adv_msfd
            else:
                print(f"[WARNING] using regression loss only for G for the first {a.freeze_step} steps")
                loss_gen_all = loss_mel

            loss_gen_all.backward()
            grad_norm_g = torch.nn.utils.clip_grad_norm_(generator.parameters(), clip_grad_norm)
            optim_g.step()

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    mel_error = (
                        loss_mel.item() / lambda_melloss
                    )  # Log training mel regression loss to stdout
                    print(
                        f"Steps: {steps:d}, "
                        f"Gen Loss Total: {loss_gen_all:4.3f}, "
                        f"Mel Error: {mel_error:4.3f}, "
                        f"s/b: {time.time() - start_b:4.3f} "
                        f"lr: {optim_g.param_groups[0]['lr']:4.7f} "
                        f"grad_norm_g: {grad_norm_g:4.3f}"
                    )

                # Checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = f"{a.checkpoint_path}/g_{steps:08d}"
                    save_checkpoint(
                        checkpoint_path,
                        {
                            "generator": (
                                generator.module if h.num_gpus > 1 else generator
                            ).state_dict()
                        },
                    )
                    checkpoint_path = f"{a.checkpoint_path}/do_{steps:08d}"
                    save_checkpoint(
                        checkpoint_path,
                        {
                            "mpd": (mpd.module if h.num_gpus > 1 else mpd).state_dict(),
                            "mrd": (mrd.module if h.num_gpus > 1 else mrd).state_dict(),
                            'msfd': (msfd.module if h.num_gpus > 1 else msfd).state_dict(),
                            "optim_g": optim_g.state_dict(),
                            "optim_d": optim_d.state_dict(),
                            "steps": steps,
                            "epoch": epoch,
                        },
                    )
                '''
                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    mel_error = (
                        loss_mel.item() / lambda_melloss
                    )  # Log training mel regression loss to tensorboard
                    sw.add_scalar("training/gen_loss_total", loss_gen_all.item(), steps)
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)
                    sw.add_scalar("training/fm_loss_mpd", loss_fm_f.item(), steps)
                    sw.add_scalar("training/gen_loss_mpd", loss_gen_f.item(), steps)
                    sw.add_scalar("training/disc_loss_mpd", loss_disc_f.item(), steps)
                    sw.add_scalar("training/grad_norm_mpd", grad_norm_mpd, steps)
                    sw.add_scalar("training/fm_loss_mrd", loss_fm_s.item(), steps)
                    sw.add_scalar("training/gen_loss_mrd", loss_gen_s.item(), steps)
                    sw.add_scalar("training/disc_loss_mrd", loss_disc_s.item(), steps)
                    sw.add_scalar("training/grad_norm_mrd", grad_norm_mrd, steps)
                    sw.add_scalar("training/grad_norm_g", grad_norm_g, steps)
                    sw.add_scalar(
                        "training/learning_rate_d", scheduler_d.get_last_lr()[0], steps
                    )
                    sw.add_scalar(
                        "training/learning_rate_g", scheduler_g.get_last_lr()[0], steps
                    )
                    sw.add_scalar("training/epoch", epoch + 1, steps)

                # validation
                if steps % a.validation_interval == 0:
                    # plot training input x so far used
                    for i_x in range(x.shape[0]):
                        sw.add_figure('training_input/x_{}'.format(i_x), plot_spectrogram(x[i_x].cpu()), steps)
                        sw.add_audio('training_input/y_{}'.format(i_x), y[i_x][0], steps, h.sampling_rate)

                    # seen and unseen speakers validation loops
                    if not a.debug and steps != 0:
                        validate(rank, a, h, validation_loader,
                                 mode="seen_{}".format(train_loader.dataset.name))
                        for i in range(len(list_unseen_validation_loader)):
                            validate(rank, a, h, list_unseen_validation_loader[i],
                                     mode="unseen_{}".format(list_unseen_validation_loader[i].dataset.name))
                '''
            steps += 1

            # BigVGAN-v2 learning rate scheduler is changed from epoch-level to step-level
            scheduler_g.step()
            scheduler_d.step()

        if rank == 0:
            print(
                f"Time taken for epoch {epoch + 1} is {int(time.time() - start)} sec\n"
            )


def main():
    print("Initializing Training Process..")

    parser = argparse.ArgumentParser()

    parser.add_argument("--group_name", default=None)

    parser.add_argument("--checkpoint_path", default="exp/bigvgan")
    parser.add_argument("--config", default="")

    parser.add_argument("--training_epochs", default=100000, type=int)
    parser.add_argument("--stdout_interval", default=5, type=int)
    parser.add_argument("--checkpoint_interval", default=50000, type=int)
    parser.add_argument("--summary_interval", default=100, type=int)
    parser.add_argument("--validation_interval", default=50000, type=int)

    parser.add_argument(
        "--freeze_step",
        default=0,
        type=int,
        help="freeze D for the first specified steps. G only uses regression loss for these steps.",
    )

    parser.add_argument("--fine_tuning", default=False, type=bool)

    parser.add_argument(
        "--debug",
        default=False,
        type=bool,
        help="debug mode. skips validation loop throughout training",
    )
    parser.add_argument(
        "--evaluate",
        default=False,
        type=bool,
        help="only run evaluation from checkpoint and exit",
    )
    parser.add_argument(
        "--eval_subsample",
        default=5,
        type=int,
        help="subsampling during evaluation loop",
    )
    parser.add_argument(
        "--skip_seen",
        default=False,
        type=bool,
        help="skip seen dataset. useful for test set inference",
    )
    parser.add_argument(
        "--save_audio",
        default=False,
        type=bool,
        help="save audio of test set inference to disk",
    )

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

    build_env(a.config, "config.json", a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print(f"Batch size per GPU: {h.batch_size}")
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(
            train,
            nprocs=h.num_gpus,
            args=(a, h,),
        )
    else:
        train(0, a, h)


if __name__ == "__main__":
    main()
