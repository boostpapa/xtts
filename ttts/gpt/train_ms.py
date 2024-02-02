import copy
from datetime import datetime
import json
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
import argparse
import logging

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


def warmup(step):
    if step < 500:
        return float(step / 500)
    else:
        return 1


class Trainer(object):
    def __init__(self, args):
        json_config = json.load(open(args.config))
        self.cfg = AttrDict(json_config)
        self.train_dataset = GptTTSDataset(self.cfg['dataset']['training_files'], self.cfg.dataset)
        self.eval_dataset = GptTTSDataset(self.cfg['dataset']['validation_files'], self.cfg.dataset)
        self.train_dataloader = DataLoader(self.train_dataset, **self.cfg['dataloader'], collate_fn=GptTTSCollater(self.cfg))
        self.eval_dataloader = DataLoader(self.eval_dataset, **self.cfg['dataloader'], collate_fn=GptTTSCollater(self.cfg))
        self.train_steps = self.cfg['train']['train_steps']
        self.eval_interval = self.cfg['train']['eval_interval']
        self.log_interval = self.cfg['train']['log_interval']
        self.num_epochs = self.cfg['train']['epochs']
        self.use_fp16 = self.cfg['train']['fp16_run']
        precision = "fp16" if self.use_fp16 else "no" # ['no', 'fp8', 'fp16', 'bf16']

        self.gpt = UnifiedVoice(**self.cfg.gpt)
        if 'pretrain_model' in self.cfg['train']:
            model_pth = self.cfg['train']['pretrain_model']
            logging.warning("loading pretrain model: {}".format(model_pth))
            gpt_checkpoint = torch.load(model_pth, map_location=torch.device("cpu"))
            gpt_checkpoint = gpt_checkpoint['model'] if 'model' in gpt_checkpoint else gpt_checkpoint
            self.gpt.load_state_dict(gpt_checkpoint, strict=False)
            print(">> GPT weights restored from:", model_pth)

        # Load DVAE
        self.dvae = DiscreteVAE(**self.cfg['vqvae'])
        self.dvae.eval()
        dvae_checkpoint = torch.load(self.cfg.dvae_checkpoint, map_location=torch.device("cpu"))
        if 'model' in dvae_checkpoint:
            dvae_checkpoint = dvae_checkpoint['model']
        self.dvae.load_state_dict(dvae_checkpoint, strict=False)

        self.accelerator = Accelerator(mixed_precision=precision, split_batches=True)
        self.model_dir = Path(args.model)
        if self.accelerator.is_main_process:
            self.model_dir.mkdir(exist_ok = True, parents=True)
        self.logger = get_logger(self.model_dir)

        self.optimizer = AdamW(self.gpt.parameters(),lr=self.cfg['train']['lr'], betas=(0.9, 0.96), weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=warmup)
        self.gpt, self.dvae, self.dataloader, self.optimizer, self.scheduler = self.accelerator.prepare(self.gpt, self.dvae, self.dataloader, self.optimizer, self.scheduler)

        self.mel_loss_weight = self.cfg['train']['mel_weight']
        self.text_loss_weight = self.cfg['train']['text_weight']
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
            data = {
                'step': self.global_step,
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
        total_losses = 0
        text_losses = 0
        mel_losses = 0
        num_samples = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.eval_dataloader):
                # speech_conditioning_latent, text_inputs, text_lengths, mel_codes, wav_lengths
                input_data = [batch['padded_raw_mel'], batch['padded_text'], batch['text_lengths'],
                              batch['padded_qmel'], batch['wav_lens']]
                input_data = [d.to(device) for d in input_data]
                loss_text, loss_mel, mel_logits = model(*input_data)
                num_sample = input_data[0].shape[0]
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
        if accelerator.is_main_process:
            writer = SummaryWriter(log_dir=self.model_dir)
            num_params = sum(p.numel() for p in self.gpt.parameters())
            print('the number of gpt model parameters: {:,d}'.format(num_params))

            self.logger.info("Initial Evaluating ...")
            losses = self.eval()
            lr = self.optimizer.param_groups[0]["lr"]
            self.logger.info([x.item() for x in losses] + [self.global_step, lr])
            self.save_checkpoint(self.model_dir.joinpath(f"init.pth"))

        for epoch in range(0, self.num_epochs):
            for batch_idx, batch in enumerate(self.train_dataloader):
                if batch is None:
                    continue
                # speech_conditioning_latent, text_inputs, text_lengths, mel_codes, wav_lengths
                input_data = [batch['padded_cond_mel'], batch['padded_text'], batch['text_lengths'],
                                batch['padded_raw_mel'], batch['wav_lens']]
                input_data = [d.to(device) for d in input_data]
                # get vqvae codes from raw mel
                input_data[3] = self.dvae.get_codebook_indices(input_data[3])
                with accelerator.autocast():
                    loss_text, loss_mel, mel_logits = self.gpt(*input_data)
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
                    lr = self.optimizer.param_groups[0]["lr"]
                    losses = [total_losses, text_losses, mel_losses]
                    self.logger.info("Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(self.train_dataloader)
                    ))
                    self.logger.info([x.item() for x in losses] + [self.global_step, lr])

                if accelerator.is_main_process and self.global_step % self.eval_interval == 0:
                    self.logger.info("Evaluating ...")
                    losses = self.eval()
                    self.logger.info([x.item() for x in losses])
                    scalar_dict = {"loss": total_losses, "loss_text": text_losses,
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
            self.scheduler.step()
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
    # trainer.load('/home/hyc/tortoise_plus_zh/ttts/gpt/logs/2023-12-24-14-22-14/model-70.pt')

    trainer.train()
