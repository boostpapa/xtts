import copy
from datetime import datetime
import json
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from ttts.utils.utils import EMA, clean_checkpoints, plot_spectrogram_to_numpy, summarize, update_moving_average
from ttts.utils.utils import get_logger
from ttts.utils.checkpoint import load_trained_modules
from ttts.vqvae.dataset import PreprocessedMelDataset
import torch
import os
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from accelerate import Accelerator
import argparse
import logging

from ttts.vqvae.xtts_dvae import DiscreteVAE

logging.getLogger("numba").setLevel(logging.WARNING)
def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def get_grad_norm(model):
    total_norm = 0
    for name,p in model.named_parameters():
        try:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        except:
            print(name)
    total_norm = total_norm ** (1. / 2) 
    return total_norm


def cycle(dl):
    while True:
        for data in dl:
            yield data


class Trainer(object):
    def __init__(self, args):
        self.cfg = json.load(open(args.config))
        self.train_dataset = PreprocessedMelDataset(self.cfg['dataset']['training_files'], self.cfg)
        self.eval_dataset = PreprocessedMelDataset(self.cfg['dataset']['validation_files'], self.cfg)
        self.train_data_loader = DataLoader(self.train_dataset, **self.cfg['dataloader'])
        self.eval_data_loader = DataLoader(self.eval_dataset, **self.cfg['dataloader'])
        self.train_steps = self.cfg['train']['train_steps']
        self.eval_interval = self.cfg['train']['eval_interval']
        self.log_interval = self.cfg['train']['log_interval']
        self.num_epochs = self.cfg['train']['epochs']
        self.c_comm = 0.25
        self.use_fp16 = self.cfg['train']['fp16_run']
        precision = "fp16" if self.use_fp16 else "no" # ['no', 'fp8', 'fp16', 'bf16'] 
        self.vqvae = DiscreteVAE(**self.cfg['vqvae'])
        if 'pretrain_model' in self.cfg['train']:
            model_pth = self.cfg['train']['pretrain_model']
            logging.warning("loading pretrain model: {}".format(model_pth))
            dvae_checkpoint = torch.load(model_pth, map_location=torch.device("cpu"))
            self.vqvae.load_state_dict(dvae_checkpoint, strict=False)
            print(">> DVAE weights restored from:", model_pth)
            #load_trained_modules(self.vqvae, model_pth)
            
        self.accelerator = Accelerator(mixed_precision=precision, split_batches=True)
        self.model_dir = Path(args.model)
        if self.accelerator.is_main_process:
            # self.ema_model = self._get_target_encoder(self.vqvae).to(self.accelerator.device)
            #now = datetime.now()
            #self.logs_folder = Path(args.model+'/'+now.strftime("%Y-%m-%d-%H-%M-%S"))
            self.model_dir.mkdir(exist_ok = True, parents=True)
        self.logger = get_logger(self.model_dir)
        #self.ema_updater = EMA(0.999)
        self.optimizer = AdamW(self.vqvae.parameters(), lr=self.cfg['train']['lr'], betas=(0.9, 0.999), weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.cfg['train']['lr_decay'])
        self.vqvae, self.train_data_loader, self.optimizer = self.accelerator.prepare(self.vqvae, self.train_data_loader, self.optimizer)
        #self.dataloader = cycle(self.dataloader)
        self.accum_grad = self.cfg['train']['accum_grad']
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
            unwrapped_model = self.accelerator.unwrap_model(self.vqvae)
            self.accelerator.save(unwrapped_model.state_dict(), path)
            #torch.save(unwrapped_model, path)

    def load_model(self, path):
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
        recon_losses = 0
        commitment_losses = 0
        ssim_losses = 0
        num_samples = 0
        with torch.no_grad():
            for batch_idx, mel in enumerate(self.eval_data_loader):
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
            writer = SummaryWriter(log_dir=self.model_dir)
            num_params = sum(p.numel() for p in self.vqvae.parameters())
            print('the number of vqvae model parameters: {:,d}'.format(num_params))

            self.logger.info("Initial Evaluating ...")
            losses = self.eval()
            self.logger.info([x.item() for x in losses])

        for epoch in range(0, self.num_epochs):
            for batch_idx, mel in enumerate(self.train_data_loader):
                recon_losses = 0
                ssim_losses = 0
                commitment_losses = 0
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
                accelerator.wait_for_everyone()

                if self.global_step % self.log_interval == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    losses = [total_losses, recon_losses, ssim_losses, commitment_losses]
                    self.logger.info("Train Epoch: {} [{:.0f}%]".format(
                            epoch, 100.0 * batch_idx / len(self.train_data_loader)
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
            self.scheduler.step()
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
