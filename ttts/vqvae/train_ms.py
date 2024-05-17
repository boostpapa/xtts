import copy
from datetime import datetime
import json
from omegaconf import OmegaConf
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from ttts.utils.utils import EMA, clean_checkpoints, plot_spectrogram_to_numpy, summarize, update_moving_average
from ttts.utils.utils import get_logger
from ttts.utils.checkpoint import load_trained_modules
from ttts.vqvae.dataset import PreprocessedMelDataset, MelCollater
import torch
import os
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from accelerate import Accelerator
from ttts.utils.utils import AttrDict, get_logger
from ttts.utils.lr_scheduler import CosineLRScheduler
import argparse
import logging

from ttts.vqvae.xtts_dvae import DiscreteVAE

logging.getLogger("numba").setLevel(logging.WARNING)


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
    total_norm = total_norm ** (1. / 2) 
    return total_norm


class Trainer(object):
    def __init__(self, args):
        if args.config.endswith(".json"):
            json_config = json.load(open(args.config))
            self.cfg = AttrDict(json_config)
        else:
            self.cfg = OmegaConf.load(args.config)
        self.train_dataset = PreprocessedMelDataset(self.cfg, self.cfg.dataset['training_files'])
        self.eval_dataset = PreprocessedMelDataset(self.cfg,  self.cfg.dataset['validation_files'], is_eval=True)
        self.train_dataloader = DataLoader(self.train_dataset, **self.cfg.dataloader, collate_fn=MelCollater(self.cfg))
        self.eval_dataloader = DataLoader(self.eval_dataset, **self.cfg.dataloader, collate_fn=MelCollater(self.cfg))
        self.train_steps = self.cfg.train['train_steps']
        self.eval_interval = self.cfg.train['eval_interval']
        self.log_interval = self.cfg.train['log_interval']
        self.num_epochs = self.cfg.train['epochs']
        self.batch_size = self.cfg.dataloader['batch_size']
        self.accum_grad = self.cfg.train['accum_grad']
        self.c_comm = 0.25
        self.lr = self.cfg.train['lr']
        self.weight_decay = self.cfg.train['weight_decay']
        self.precision = self.cfg.train['precision']
        # ['no', 'fp8', 'fp16', 'bf16']
        precision = self.precision
        if self.precision == "fp32":
            precision = "no"
        print(">> training precision:", precision)

        self.vqvae = DiscreteVAE(**self.cfg['vqvae'])
        if 'pretrain_model' in self.cfg['train']:
            model_pth = self.cfg.train['pretrain_model']
            logging.warning("loading pretrain model: {}".format(model_pth))
            dvae_checkpoint = torch.load(model_pth, map_location=torch.device("cpu"))
            dvae_checkpoint = dvae_checkpoint['model'] if 'model' in dvae_checkpoint else dvae_checkpoint
            self.vqvae.load_state_dict(dvae_checkpoint, strict=False)
            print(">> DVAE weights restored from:", model_pth)
            
        self.accelerator = Accelerator(mixed_precision=precision, split_batches=True)
        self.model_dir = Path(args.model)
        if self.accelerator.is_main_process:
            self.model_dir.mkdir(exist_ok=True, parents=True)
        self.logger = get_logger(self.model_dir)

        self.optimizer = AdamW(self.vqvae.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=self.weight_decay)
        total_batches = len(self.train_dataloader)
        total_training_steps = total_batches * self.num_epochs / self.accum_grad
        print(f">> total training epoch: {self.num_epochs}, batches per epoch: {total_batches}, steps: {total_training_steps}")
        if 'min_lr' in self.cfg.train:
            self.min_lr = self.cfg.train['min_lr']
            num_warmup_step = self.cfg.train['warmup_steps']
            final_lr_ratio = self.min_lr / self.lr

        #self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.cfg['train']['lr_decay'])
        self.scheduler = CosineLRScheduler(self.optimizer, warmup_steps=num_warmup_step,
                                           total_steps=total_training_steps, lr_min_ratio=final_lr_ratio)
        self.vqvae, self.train_dataloader, self.optimizer, self.scheduler = self.accelerator.prepare(self.vqvae, self.train_dataloader, self.optimizer, self.scheduler)

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
                'model': self.accelerator.get_state_dict(self.vqvae),
            }
            torch.save(data, path)
            #unwrapped_model = self.accelerator.unwrap_model(self.vqvae)
            #self.accelerator.save(unwrapped_model.state_dict(), path)
            #torch.save(unwrapped_model, path)

    def load_model(self, model_path):
        accelerator = self.accelerator
        device = accelerator.device
        data = torch.load(model_path, map_location=device)
        state_dict = data['model']
        self.global_step = data['step']
        vqvae = accelerator.unwrap_model(self.vqvae)
        vqvae.load_state_dict(state_dict)
        # if self.accelerator.is_local_main_process:
        #     self.ema_model.load_state_dict(state_dict)

    def eval(self):
        model = self.accelerator.unwrap_model(self.vqvae)
        device = self.accelerator.device
        model.eval()
        recon_losses = 0.0
        commitment_losses = 0.0
        ssim_losses = 0.0
        num_samples = 0.0
        with torch.no_grad():
            for batch_idx, mel in enumerate(self.eval_dataloader):
                mel = mel.to(device).squeeze(1)
                recon_loss, ssim_loss, commitment_loss, mel_recon = model(mel)
                #recon_loss = torch.mean(recon_loss, dim=(1, 2))
                #recon_loss = torch.sum(recon_loss)
                num_sample = mel.shape[0]
                recon_losses += recon_loss * num_sample
                ssim_losses += ssim_loss * num_sample
                commitment_losses += commitment_loss * num_sample
                num_samples += num_sample

        model.train()
        recon_losses /= num_samples
        ssim_losses /= num_samples
        commitment_losses /= num_samples
        total_losses = recon_losses + ssim_losses + self.c_comm * commitment_losses
        return [total_losses, recon_losses, ssim_losses, commitment_losses]

    def train(self):
        accelerator = self.accelerator
        device = self.accelerator.device
        if accelerator.is_main_process:
            self.logger.info(self.cfg)
            writer = SummaryWriter(log_dir=self.model_dir)
            print(self.vqvae)
            num_params = sum(p.numel() for p in self.vqvae.parameters())
            print('the number of vqvae model parameters: {:,d}'.format(num_params))

            self.logger.info("Initial Evaluating ...")
            losses = self.eval()
            lr = self.optimizer.param_groups[0]["lr"]
            self.logger.info([x.item() for x in losses] + [self.global_step, lr])
            self.save_checkpoint(self.model_dir.joinpath(f"init.pth"))

        for epoch in range(0, self.num_epochs):
            for batch_idx, mel in enumerate(self.train_dataloader):
                recon_losses = 0.0
                ssim_losses = 0.0
                commitment_losses = 0.0
                total_losses = 0.
                for _ in range(self.accum_grad):
                    mel = mel.to(device).squeeze(1)
                    with accelerator.autocast():
                        recon_loss, ssim_loss, commitment_loss, mel_recon = self.vqvae(mel)
                        #recon_loss = torch.mean(recon_loss, dim=(1, 2))
                        #recon_loss = torch.mean(recon_loss)
                        loss = recon_loss + ssim_loss + self.c_comm * commitment_loss
                        loss = loss / self.accum_grad
                    accelerator.backward(loss)
                    total_losses += loss
                    recon_losses += recon_loss / self.accum_grad
                    ssim_losses += ssim_loss / self.accum_grad
                    commitment_losses += commitment_loss / self.accum_grad

                grad_norm = get_grad_norm(self.vqvae)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(self.vqvae.parameters(), self.grad_clip)
                accelerator.wait_for_everyone()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
                accelerator.wait_for_everyone()

                if self.global_step % self.log_interval == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    losses = [total_losses, recon_losses, ssim_losses, commitment_losses]
                    self.logger.info("Train Epoch: {} [{:.0f}%]".format(
                            epoch, 100.0 * batch_idx / len(self.train_dataloader)
                        ))
                    self.logger.info([x.item() for x in losses] + [self.global_step, lr])

                if accelerator.is_main_process and self.global_step % self.eval_interval == 0:
                    self.logger.info("Evaluating ...")
                    losses = self.eval()
                    self.logger.info([x.item() for x in losses])
                 
                    '''
                    keep_ckpts = self.cfg['train']['keep_ckpts']
                    if keep_ckpts > 0:
                        clean_checkpoints(path_to_models=self.model_dir,
                                          n_ckpts_to_keep=keep_ckpts, sort_by_time=True)
                    self.save_checkpoint(self.model_dir.joinpath(f"model_{self.global_step}.pth"))
                    '''
                    scalar_dict = {"loss": total_losses, "loss_mel": recon_losses, "loss_commitment": commitment_losses,
                                   "loss/grad": grad_norm}
                    summarize(
                        writer=writer,
                        global_step=self.global_step,
                        scalars=scalar_dict
                    )
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
    parser = argparse.ArgumentParser(description='train vqvae')
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
    #args = parser.parse_args()
    args, _ = parser.parse_known_args()

    return args


if __name__ == '__main__':
    args = get_args()
    print(args)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    trainer = Trainer(args)
    # trainer.load('~/tortoise_plus_zh/ttts/vqvae/logs/2023-11-04-00-25-39/model-14.pt')
     
    trainer.train()
