from omegaconf import OmegaConf
import torch
import random
import time
from datetime import datetime
import json
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator, DistributedDataParallelKwargs

from ttts.flow_matching_dit.dataset import FlowMatchingDataset, FlowMatchingCollator
from ttts.utils.utils import AttrDict, get_logger
from ttts.utils.utils import summarize
from ttts.flow_matching_dit.model import StableTTS
from ttts.utils.lr_scheduler import CosineLRScheduler
from ttts.utils.utils import make_pad_mask
from setproctitle import setproctitle
from ttts.utils.checkpoint import load_checkpoint, load_pretrain_modules
from ttts.vqvae.xtts_dvae import DiscreteVAE
from ttts.gpt.model import UnifiedVoice
import argparse
import logging

logging.getLogger("numba").setLevel(logging.WARNING)


num_warmup_step = 1000
total_training_steps = 100000
final_lr_ratio = 0.1


class Trainer(object):
    def __init__(self, args):
        # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        if args.config.endswith(".json"):
            json_config = json.load(open(args.config))
            self.cfg = AttrDict(json_config)
        else:
            self.cfg = OmegaConf.load(args.config)

        self.train_dataset = FlowMatchingDataset(self.cfg, self.cfg.dataset['training_files'])
        self.eval_dataset = FlowMatchingDataset(self.cfg, self.cfg.dataset['validation_files'], is_eval=True)
        self.train_dataloader = DataLoader(self.train_dataset, **self.cfg.dataloader,
                                           collate_fn=FlowMatchingCollator())
        self.eval_dataloader = DataLoader(self.eval_dataset, **self.cfg.dataloader,
                                          collate_fn=FlowMatchingCollator())

        self.train_steps = self.cfg.train['train_steps']
        self.eval_interval = self.cfg.train['eval_interval']
        self.save_interval = self.cfg.train['save_interval'] if 'save_interval' in self.cfg.train else None
        self.log_interval = self.cfg.train['log_interval']
        self.num_epochs = self.cfg.train['epochs']
        self.accum_grad = self.cfg.train['accum_grad']
        self.lr = self.cfg.train['lr']
        self.precision = self.cfg.train['precision']
        # ['no', 'fp8', 'fp16', 'bf16']
        precision = self.precision
        if self.precision == "fp32":
            precision = "no"
        print(">> training precision:", precision)

        self.global_step = 0
        self.start_epoch = 0

        self.flow_matching = StableTTS(self.cfg["flow_matching_dit"])

        if 'checkpoint' in self.cfg.train:
            model_pth = self.cfg.train['checkpoint']
            self.global_step, self.start_epoch = load_checkpoint(self.flow_matching, model_pth)
            print(">> Flow matching weights restored from checkpoint:", model_pth)
        elif 'pretrain_model' in self.cfg.train:
            model_pth = self.cfg.train['pretrain_model']
            load_pretrain_modules(self.flow_matching, model_pth)
            print(">> Flow matching weights initialize with pretrain model:", model_pth)
        if 'step' in self.cfg.train:
            self.global_step = self.cfg.train['step']
        if 'start_epoch' in self.cfg.train:
            self.start_epoch = self.cfg.train['start_epoch']

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

        # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        self.accelerator = Accelerator(mixed_precision=precision, split_batches=True)
        self.model_dir = Path(args.model)
        if self.accelerator.is_main_process:
            self.model_dir.mkdir(exist_ok=True, parents=True)
        self.logger = get_logger(self.model_dir)

        self.optimizer = AdamW(self.flow_matching.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.01)
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

        self.scheduler = CosineLRScheduler(self.optimizer, warmup_steps=num_warmup_step, total_steps=total_training_steps, lr_min_ratio=final_lr_ratio)
        self.scheduler.set_step(self.global_step)
        self.flow_matching, self.train_dataloader, self.eval_dataloader, self.optimizer, self.scheduler, self.gpt, self.dvae \
            = self.accelerator.prepare(self.flow_matching, self.train_dataloader, self.eval_dataloader, self.optimizer, self.scheduler, self.gpt, self.dvae)
        self.grad_clip = self.cfg.train['grad_clip']
        if self.grad_clip <= 0:
            self.grad_clip = 50
        segment_size = self.cfg.bigvgan.segment_size  # 11264 # 8192 # 24576
        hop_length = self.cfg.bigvgan.gpt_dim
        self.chunk = segment_size // hop_length

    def eval(self):
        model = self.accelerator.unwrap_model(self.flow_matching)
        device = self.accelerator.device
        model.eval()
        total_losses = 0.
        num_samples = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.eval_dataloader):
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

                with torch.no_grad():
                    mel_code = self.dvae.get_codebook_indices(mel_infer)
                    # latent = self.gpt(mel_refer,
                    latent, text_lens_out, code_lens_out = self.gpt(mel_refer,
                                                                    text,
                                                                    text_lens,
                                                                    mel_code,
                                                                    wav_infer_lens,
                                                                    cond_mel_lengths=mel_refer_len,
                                                                    return_latent=True,
                                                                    clip_inputs=False,)
                    latent = latent.transpose(1, 2)
                    latent_list = []
                    for lat, t_len in zip(latent, text_lens_out):
                        lat = lat[:, t_len:]
                        latent_list.append(lat)

                    x = []
                    y = []
                    y_len = []
                    # print(f"code_lens_out {code_lens_out}, mel_infer_len {mel_infer_len}")
                    for lat, mel, len_ in zip(latent_list, mel_infer, mel_infer_len):
                        # [T], [1024, T/1024], 1
                        start = 0
                        if len_ // 4 > self.chunk:
                            start = random.randint(0, len_ // 4 - self.chunk)
                            y_len.append(self.chunk*4)
                        else:
                            y_len.append(len_)
                        gpt_latent = lat[:, start:start + self.chunk]
                        mel = mel[:, start * 4: (start + self.chunk) * 4]
                        # print(f"lat shape {lat.shape}, gpt_latent shape {gpt_latent.shape}")
                        x.append(gpt_latent)
                        y.append(mel)
                    latent = torch.stack(x)
                    mel = torch.stack(y)
                    mel_len = torch.LongTensor(y_len)
                    #print(f"latent shape {latent.shape}, mel shape {mel.shape}, mel_len {mel_len}")

                with self.accelerator.autocast():
                    loss = self.flow_matching(
                        x=latent,
                        y=mel,
                        y_lengths=mel_len,
                        z=mel_refer,
                        z_lengths=mel_refer_len,
                    )
                    loss = loss / self.accum_grad
                num_sample = mel_code.shape[0]
                num_samples += num_sample
                total_losses += loss * num_sample

        model.train()
        total_losses /= num_samples
        return total_losses

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        setproctitle("test_flow_matching")
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

            print(self.flow_matching)
            num_params = sum(p.numel() for p in self.flow_matching.parameters())
            print('the number of flow matching model parameters: {:,d}'.format(num_params))

            self.logger.info("Initial Evaluating ...")
            losses = self.eval()
            lr = self.optimizer.param_groups[0]["lr"]
            self.logger.info([x.item() for x in losses] + [self.global_step, lr])
            #self.save_checkpoint(self.model_dir.joinpath(f"init.pth"), lr, self.start_epoch, self.global_step)

        for epoch in range(self.start_epoch, self.num_epochs):
            total_losses = 0.
            for batch_idx, batch in enumerate(self.train_dataloader):
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

                with torch.no_grad():
                    mel_code = self.dvae.get_codebook_indices(mel_infer)
                    # latent = self.gpt(mel_refer,
                    latent, text_lens_out, code_lens_out = self.gpt(mel_refer,
                                                                    text,
                                                                    text_lens,
                                                                    mel_code,
                                                                    wav_infer_lens,
                                                                    cond_mel_lengths=mel_refer_len,
                                                                    return_latent=True,
                                                                    clip_inputs=False,)
                    latent = latent.transpose(1, 2)
                    latent_list = []
                    for lat, t_len in zip(latent, text_lens_out):
                        lat = lat[:, t_len:]
                        latent_list.append(lat)

                    x = []
                    y = []
                    y_len = []
                    # print(f"code_lens_out {code_lens_out}, mel_infer_len {mel_infer_len}")
                    for lat, mel, len_ in zip(latent_list, mel_infer, mel_infer_len):
                        # [T], [1024, T/1024], 1
                        start = 0
                        if len_ // 4 > self.chunk:
                            start = random.randint(0, len_ // 4 - self.chunk)
                            y_len.append(self.chunk * 4)
                        else:
                            y_len.append(len_)
                        gpt_latent = lat[:, start:start + self.chunk]
                        mel = mel[:, start * 4: (start + self.chunk) * 4]
                        # print(f"lat shape {lat.shape}, gpt_latent shape {gpt_latent.shape}")
                        x.append(gpt_latent)
                        y.append(mel)
                    latent = torch.stack(x)
                    mel = torch.stack(y)
                    mel_len = torch.LongTensor(y_len)
                    # print(f"latent shape {latent.shape}, mel shape {mel.shape}, mel_len {mel_len}")

                with self.accelerator.autocast():
                    loss = self.flow_matching(
                                x=latent,
                                y=mel,
                                y_lengths=mel_len,
                                z=mel_refer,
                                z_lengths=mel_refer_len,)
                    loss = loss / self.accum_grad
                accelerator.backward(loss)
                total_losses += loss

                if batch_idx % self.accum_grad != 0:
                    continue

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(self.flow_matching.parameters(), self.grad_clip)
                accelerator.wait_for_everyone()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
                accelerator.wait_for_everyone()

                if self.global_step % self.log_interval == 0:
                    #logging.warning(f"batch size: {input_data[3].shape}")
                    lr = self.optimizer.param_groups[0]["lr"]
                    self.logger.info("Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(self.train_dataloader)
                    ))
                    self.logger.info(
                        f"Train Epoch: {epoch:d}, "
                        f"{100.0 * batch_idx / len(self.train_dataloader):.0f}%, "
                        f"Steps: {self.global_step:d}, "
                        f"MSE Error: {total_losses:4.3f}, "
                        f"s/b: {time.time() - start_b:4.3f} "
                        f"lr: {lr:4.7f} "
                        f"grad_norm: {grad_norm:4.3f}"
                    )

                if accelerator.is_main_process and batch_idx > 0 and self.save_interval is not None and self.global_step % self.save_interval == 0:
                    self.logger.info("Saving checkpoint ...")
                    self.save_checkpoint(self.model_dir.joinpath(f"checkpoint_{self.global_step}.pth"), lr, epoch, self.global_step)

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
                total_losses = self.eval()
                lr = self.optimizer.param_groups[0]["lr"]
                self.logger.info(
                    f"Evaluating Epoch: {epoch:d}, "
                    f"Steps: {self.global_step:d}, "
                    f"MSE Error: {total_losses:4.3f}, "
                    f"lr: {lr:4.7f} "
                )
                self.save_checkpoint(self.model_dir.joinpath(f"epoch_{epoch}.pth"), lr, epoch, self.global_step)
        accelerator.print('training complete')


def get_args():
    parser = argparse.ArgumentParser(description='train flow matching')
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

    trainer.train()
