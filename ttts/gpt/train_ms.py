import copy
from datetime import datetime
import json, math
from omegaconf import OmegaConf
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from ttts.utils.utils import EMA, clean_checkpoints, plot_spectrogram_to_numpy, summarize, update_moving_average
from ttts.gpt.dataset import GptTTSCollater, GptTTSDataset
from ttts.gpt.model import UnifiedVoice
from ttts.vqvae.xtts_dvae import DiscreteVAE
import torch
import os
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import AdamW
from accelerate import Accelerator
from ttts.utils.utils import AttrDict, get_logger
from ttts.utils.lr_scheduler import CosineLRScheduler
import argparse
import logging
from setproctitle import setproctitle
from ttts.utils.checkpoint import load_checkpoint, load_pretrain_modules

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


num_warmup_step = 1000
total_training_steps = 100000
final_lr_ratio = 0.1


def get_cosine_schedule_with_warmup_lr(
    current_step: int,
):
    global num_warmup_step
    global total_training_steps
    global final_lr_ratio
    if current_step < num_warmup_step:
        return float(current_step) / float(max(1, num_warmup_step))

    progress = float(current_step - num_warmup_step) / float(
        max(1, total_training_steps - num_warmup_step)
    )

    lr_ratio = 0.5 * (1.0 + math.cos(math.pi * progress))
    return max(final_lr_ratio, lr_ratio)


class Trainer(object):
    def __init__(self, args):
        if args.config.endswith(".json"):
            json_config = json.load(open(args.config))
            self.cfg = AttrDict(json_config)
        else:
            self.cfg = OmegaConf.load(args.config)
        self.train_dataset = GptTTSDataset(self.cfg, self.cfg.dataset['training_files'], is_eval=False)
        self.eval_dataset = GptTTSDataset(self.cfg, self.cfg.dataset['validation_files'], is_eval=True)
        self.train_dataloader = DataLoader(self.train_dataset, **self.cfg.dataloader, collate_fn=GptTTSCollater(self.cfg))
        self.eval_dataloader = DataLoader(self.eval_dataset, **self.cfg.dataloader_eval, collate_fn=GptTTSCollater(self.cfg))
        self.train_steps = self.cfg.train['train_steps']
        self.eval_interval = self.cfg.train['eval_interval']
        self.save_interval = self.cfg.train['save_interval'] if 'save_interval' in self.cfg.train else None
        self.log_interval = self.cfg.train['log_interval']
        self.num_epochs = self.cfg.train['epochs']
        self.batch_size = self.cfg.dataloader['batch_size']
        self.accum_grad = self.cfg.train['accum_grad']
        self.lr = self.cfg.train['lr']
        self.weight_decay = self.cfg.train['weight_decay']
        self.precision = self.cfg.train['precision']
        # ['no', 'fp8', 'fp16', 'bf16']
        precision = self.precision
        if self.precision == "fp32":
            precision = "no"
        print(">> training precision:", precision)

        self.global_step = 0
        self.start_epoch = 0
        self.gpt = UnifiedVoice(**self.cfg.gpt)
        if 'checkpoint' in self.cfg.train:
            model_pth = self.cfg.train['checkpoint']
            self.global_step, self.start_epoch = load_checkpoint(self.gpt, model_pth)
            print(">> GPT weights restored from checkpoint:", model_pth)
        elif 'pretrain_model' in self.cfg.train:
            model_pth = self.cfg.train['pretrain_model']
            load_pretrain_modules(self.gpt, model_pth)
            print(">> GPT weights initialize with pretrain model:", model_pth)
        if 'step' in self.cfg.train:
            self.global_step = self.cfg.train['step']
        if 'start_epoch' in self.cfg.train:
            self.start_epoch = self.cfg.train['start_epoch']

        # Load DVAE
        self.dvae = DiscreteVAE(**self.cfg['vqvae'])
        dvae_checkpoint = torch.load(self.cfg.dvae_checkpoint, map_location=torch.device("cpu"))
        if 'model' in dvae_checkpoint:
            dvae_checkpoint = dvae_checkpoint['model']
        self.dvae.load_state_dict(dvae_checkpoint, strict=False)

        self.accelerator = Accelerator(mixed_precision=precision, split_batches=True)
        self.model_dir = Path(args.model)
        if self.accelerator.is_main_process:
            self.model_dir.mkdir(exist_ok=True, parents=True)
        self.logger = get_logger(self.model_dir)

        self.optimizer = AdamW(self.gpt.parameters(), lr=self.lr, betas=(0.9, 0.96), weight_decay=self.weight_decay)
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

        #self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=get_cosine_schedule_with_warmup_lr)
        self.scheduler = CosineLRScheduler(self.optimizer, warmup_steps=num_warmup_step, total_steps=total_training_steps, lr_min_ratio=final_lr_ratio)
        self.scheduler.set_step(self.global_step)
        self.gpt, self.dvae, self.train_dataloader, self.eval_dataloader, self.optimizer, self.scheduler = self.accelerator.prepare(self.gpt, self.dvae, self.train_dataloader, self.eval_dataloader, self.optimizer, self.scheduler)
        self.dvae.eval()

        self.mel_loss_weight = self.cfg.train['mel_weight']
        self.text_loss_weight = self.cfg.train['text_weight']
        self.grad_clip = self.cfg.train['grad_clip']
        if self.grad_clip <= 0:
            self.grad_clip = 50

    def _get_target_encoder(self, model):
        target_encoder = copy.deepcopy(model)
        set_requires_grad(target_encoder, False)
        for p in target_encoder.parameters():
            p.DO_NOT_TRAIN = True
        return target_encoder

    def save_checkpoint(self, path, lr, epoch, step):
        if self.accelerator.is_main_process:
            data = {
                'lr': lr,
                'epoch': epoch,
                'step': step,
                'model': self.accelerator.get_state_dict(self.gpt),
            }
            torch.save(data, path)
            # unwrapped_model = self.accelerator.unwrap_model(self.gpt)
            # self.accelerator.save(unwrapped_model.state_dict(), path)
            # torch.save(unwrapped_model, path)

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return
        data = {
            'step': self.global_step,
            'model': self.accelerator.get_state_dict(self.gpt),
        }
        torch.save(data, str(self.logs_folder / f'model-{milestone}.pt'))

    def load(self, model_path):
        accelerator = self.accelerator
        device = accelerator.device
        data = torch.load(model_path, map_location=device)
        state_dict = data['model']
        self.global_step = data['step']
        gpt = accelerator.unwrap_model(self.gpt)
        gpt.load_state_dict(state_dict)
        if self.accelerator.is_local_main_process:
            self.ema_model.load_state_dict(state_dict)

    def eval(self):
        model = self.accelerator.unwrap_model(self.gpt)
        device = self.accelerator.device
        model.eval()
        text_losses = mel_losses = total_losses = 0.
        num_samples = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.eval_dataloader):
                # speech_conditioning_latent, text_inputs, text_lengths, mel_codes, wav_lengths
                input_data = [batch['padded_cond_mel'], batch['padded_text'], batch['text_lengths'],
                                batch['padded_raw_mel'], batch['wav_lens']]
                input_data = [d.to(device) for d in input_data]
                # get vqvae codes from raw mel
                input_data[3] = self.dvae.get_codebook_indices(input_data[3])
                loss_text, loss_mel, mel_logits = model(*input_data, cond_mel_lengths=batch['cond_mel_lengths'])
                num_sample = input_data[0].shape[0]
                #self.logger.info([loss_text, loss_mel])
                text_losses += loss_text * num_sample
                mel_losses += loss_mel * num_sample
                num_samples += num_sample

        model.train()
        text_losses /= num_samples
        mel_losses /= num_samples
        total_losses = text_losses * self.text_loss_weight + mel_losses * self.mel_loss_weight
        return [total_losses, text_losses, mel_losses]

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        setproctitle("test_xtts_gpt")
        if isinstance(self.dvae, torch.nn.parallel.DistributedDataParallel):
            self.dvae = self.dvae.module

        if accelerator.is_main_process:
            self.logger.info(self.cfg)
            writer = SummaryWriter(log_dir=self.model_dir)
            num_params = sum(p.numel() for p in self.gpt.parameters())
            print(self.gpt)
            print('the number of gpt model parameters: {:,d}'.format(num_params))

            self.logger.info("Initial Evaluating ...")
            losses = self.eval()
            lr = self.optimizer.param_groups[0]["lr"]
            self.logger.info([x.item() for x in losses] + [self.global_step, lr])
            #self.save_checkpoint(self.model_dir.joinpath(f"init.pth"), lr, self.start_epoch, self.global_step)

        for epoch in range(self.start_epoch, self.num_epochs):
            text_losses = mel_losses = total_losses = 0.
            for batch_idx, batch in enumerate(self.train_dataloader):
                if batch is None:
                    continue
                # speech_conditioning_latent, text_inputs, text_lengths, mel_codes, wav_lengths
                input_data = [batch['padded_cond_mel'], batch['padded_text'], batch['text_lengths'],
                                batch['padded_raw_mel'], batch['wav_lens']]
                input_data = [d.to(device) for d in input_data]
                # get vqvae codes from raw mel
                with torch.no_grad():
                    input_data[3] = self.dvae.get_codebook_indices(input_data[3])
                with accelerator.autocast():
                    loss_text, loss_mel, mel_logits = self.gpt(*input_data, cond_mel_lengths=batch['cond_mel_lengths'])
                    loss = loss_text * self.text_loss_weight + loss_mel * self.mel_loss_weight
                    loss = loss / self.accum_grad
                accelerator.backward(loss)
                total_losses += loss
                text_losses += loss_text / self.accum_grad
                mel_losses += loss_mel / self.accum_grad

                if batch_idx % self.accum_grad != 0:
                    continue

                grad_norm = get_grad_norm(self.gpt)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(self.gpt.parameters(), self.grad_clip)
                accelerator.wait_for_everyone()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
                accelerator.wait_for_everyone()

                if self.global_step % self.log_interval == 0:
                    #logging.warning(f"batch size: {input_data[3].shape}")
                    lr = self.optimizer.param_groups[0]["lr"]
                    losses = [total_losses, text_losses, mel_losses]
                    self.logger.info("Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(self.train_dataloader)
                    ))
                    self.logger.info([x.item() for x in losses] + [self.global_step, lr])

                if accelerator.is_main_process and batch_idx > 0 and self.save_interval is not None and self.global_step % self.save_interval == 0:
                    self.logger.info("Saving checkpoint ...")
                    self.save_checkpoint(self.model_dir.joinpath(f"checkpoint_{self.global_step}.pth"), lr, epoch, self.global_step)

                if accelerator.is_main_process and self.global_step % self.eval_interval == 0:
                    self.logger.info("Evaluating ...")
                    losses = self.eval()
                    self.logger.info([x.item() for x in losses])
                    # self.save_checkpoint(self.model_dir.joinpath(f"checkpoint_{self.global_step}.pth"), lr, epoch, self.global_step)
                    scalar_dict = {"loss": total_losses,
                                    "loss_text": text_losses,
                                    "loss_mel": mel_losses,
                                    "loss/grad": grad_norm}
                    summarize(
                        writer=writer,
                        global_step=self.global_step,
                        scalars=scalar_dict
                    )
                text_losses = mel_losses = total_losses = 0.
                self.global_step += 1
            # one epoch training finish
            if accelerator.is_main_process:
                self.logger.info(f"Evaluating Epoch: {epoch}")
                losses = self.eval()
                lr = self.optimizer.param_groups[0]["lr"]
                self.logger.info([x.item() for x in losses] + [self.global_step, lr])
            self.save_checkpoint(self.model_dir.joinpath(f"epoch_{epoch}.pth"), lr, epoch, self.global_step)
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
    # trainer.load('/home/hyc/tortoise_plus_zh/ttts/gpt/logs/2023-12-24-14-22-14/model-70.pt')

    trainer.train()
