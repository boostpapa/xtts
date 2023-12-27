import copy
from datetime import datetime
import json
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from ttts.utils.utils import EMA, clean_checkpoints, plot_spectrogram_to_numpy, summarize, update_moving_average
from ttts.vqvae.dataset import PreprocessedMelDataset
import torch
import os
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import AdamW
from accelerate import Accelerator
import argparse
import logging

from ttts.vqvae.xtts_dvae import DiscreteVAE


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
        self.accelerator = Accelerator()
        self.cfg = json.load(open(args.config))
        self.vqvae = DiscreteVAE(**self.cfg['vqvae'])
        self.train_dataset = PreprocessedMelDataset(self.cfg['dataset']['training_files'], self.cfg)
        self.eval_dataset = PreprocessedMelDataset(self.cfg['dataset']['validation_files'], self.cfg)
        self.train_data_loader = DataLoader(self.train_dataset, **self.cfg['dataloader'])
        self.eval_data_loader = DataLoader(self.eval_dataset, **self.cfg['dataloader'])
        self.train_steps = self.cfg['train']['train_steps']
        self.eval_interval = self.cfg['train']['eval_interval']
        self.log_interval = self.cfg['train']['log_interval']
        self.num_epochs = self.cfg['train']('epochs')
        self.c_comm = 0.25
        if self.accelerator.is_main_process:
            # self.ema_model = self._get_target_encoder(self.vqvae).to(self.accelerator.device)
            #now = datetime.now()
            #self.logs_folder = Path(args.model+'/'+now.strftime("%Y-%m-%d-%H-%M-%S"))
            self.model_dir = Path(args.model)
            self.model_dir.mkdir(exist_ok = True, parents=True)
        #self.ema_updater = EMA(0.999)
        self.optimizer = AdamW(self.vqvae.parameters(), lr=3e-4, betas=(0.9, 0.9999), weight_decay=0.01)
        self.vqvae, self.train_data_loader, self.optimizer = self.accelerator.prepare(self.vqvae, self.train_data_loader, self.optimizer)
        #self.dataloader = cycle(self.dataloader)
        self.global_step = 0
        self.gradient_accumulate_every = 1

    def _get_target_encoder(self, model):
        target_encoder = copy.deepcopy(model)
        set_requires_grad(target_encoder, False)
        for p in target_encoder.parameters():
            p.DO_NOT_TRAIN = True
        return target_encoder

    def save_checkpoint(self, filepath):
        if not self.accelerator.is_local_main_process:
            return
        data = {
            'step': self.global_step,
            'model': self.accelerator.get_state_dict(self.vqvae),
        }
        torch.save(data, filepath)

    def load(self, model_path):
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
        recon_losses = 0
        commitment_losses = 0
        total_losses = 0
        num_samples = 0
        with torch.no_grad():
            eval_model = self.accelerator.unwrap_model(self.vqvae)
            eval_model.eval()
            for batch_idx, mel in enumerate(self.eval_data_loader):
                batch_size = mel.size(0)
                recon_loss, commitment_loss, mel_recon = eval_model(mel)
                recon_losses += recon_loss.item()
                commitment_losses += commitment_loss.item()
                num_samples += batch_size
            eval_model.train()

        recon_losses /= num_samples
        commitment_losses /= len(self.eval_data_loader)
        total_losses = recon_losses + self.c_comm * commitment_losses
        return [total_losses, recon_losses, commitment_losses]

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        if accelerator.is_main_process:
            writer = SummaryWriter(log_dir=self.model_dir)

        for epoch in range(0, self.num_epochs):
            for batch_idx, mel in enumerate(self.train_data_loader):
                recon_losses = 0
                commitment_losses = 0
                total_losses = 0.
                for _ in range(self.gradient_accumulate_every):
                    mel = mel.to(device).squeeze(1)
                    with self.accelerator.autocast():
                        recon_loss, commitment_loss, mel_recon = self.vqvae(mel)
                        recon_loss = torch.mean(recon_loss)
                        loss = recon_loss + 0.25 * commitment_loss
                        loss = loss / self.gradient_accumulate_every
                    self.accelerator.backward(loss)
                    total_losses += loss.item()
                    recon_losses += recon_loss.item()
                    commitment_losses += commitment_loss.item()

                grad_norm = get_grad_norm(self.vqvae)
                accelerator.clip_grad_norm_(self.vqvae.parameters(), 5.0)
                accelerator.wait_for_everyone()
                self.optimizer.step()
                self.optimizer.zero_grad()
                accelerator.wait_for_everyone()

            if accelerator.is_main_process and self.global_step % self.log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                losses = [total_losses, recon_losses, commitment_losses]
                logging.info("Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(self.train_data_loader)
                    ))
                logging.info([x.item() for x in losses] + [self.global_step, lr])

            if accelerator.is_main_process and self.global_step % self.eval_interval == 0:
                logging.info("Evaluating ...")
                losses = eval()
                logging.info(losses)

                keep_ckpts = self.cfg['train']['keep_ckpts']
                if keep_ckpts > 0:
                    clean_checkpoints(path_to_models=self.model_dir,
                                      n_ckpts_to_keep=keep_ckpts, sort_by_time=True)
                self.save_checkpoint(self.model_dir.joinpath(f"model_{self.global_step}.pth"))

                scalar_dict = {"loss": total_losses, "loss_mel": recon_losses, "loss_commitment": commitment_losses,
                               "loss/grad": grad_norm}
                summarize(
                    writer=writer,
                    global_step=self.global_step,
                    scalars=scalar_dict
                )
            self.global_step += 1
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
