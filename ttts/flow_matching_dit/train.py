import os
import sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
from omegaconf import OmegaConf
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from dataclasses import asdict

from ttts.flow_matching_dit.dataset import MatchingFlowDataset, MatchingFlowCollator
from ttts.utils.utils import summarize, get_grad_norm
from ttts.flow_matching_dit.model import StableTTS, normalize_tacotron_mel
from ttts.flow_matching_dit.scheduler import get_cosine_schedule_with_warmup
from ttts.utils.utils import get_logger
from ttts.utils.lr_scheduler import CosineLRScheduler
from ttts.utils.checkpoint import load_checkpoint, save_checkpoint
from ttts.vqvae.xtts_dvae import DiscreteVAE
import argparse
import logging


torch.backends.cudnn.benchmark = True
def continue_training(checkpoint_path, model: DDP, optimizer: optim.Optimizer) -> int:
    """load the latest checkpoints and optimizers"""
    model_dict = {}
    optimizer_dict = {}
    
    # globt all the checkpoints in the directory
    for file in os.listdir(checkpoint_path):
        if file.endswith(".pt") and '_' in file:
            name, epoch_str = file.rsplit('_', 1)
            epoch = int(epoch_str.split('.')[0])
            
            if name.startswith("checkpoint"):
                model_dict[epoch] = file
            elif name.startswith("optimizer"):
                optimizer_dict[epoch] = file
    
    # get the largest epoch
    common_epochs = set(model_dict.keys()) & set(optimizer_dict.keys())
    if common_epochs:
        max_epoch = max(common_epochs)
        model_path = os.path.join(checkpoint_path, model_dict[max_epoch])
        optimizer_path = os.path.join(checkpoint_path, optimizer_dict[max_epoch])
        
        # load model and optimizer
        model.module.load_state_dict(torch.load(model_path, map_location='cpu'))
        optimizer.load_state_dict(torch.load(optimizer_path, map_location='cpu'))
        
        print(f'resume model and optimizer from {max_epoch} epoch')
        return max_epoch + 1
   
    else:
        # load pretrained checkpoint
        if model_dict:
            model_path = os.path.join(checkpoint_path, model_dict[max(model_dict.keys())])
            model.module.load_state_dict(torch.load(model_path, map_location='cpu'))
            
        return 0


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group("gloo" if os.name == "nt" else "nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
def _init_config(config_file, outdir):
    os.makedirs(os.path.join(outdir, "checkpoint"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "summary"), exist_ok=True)
    
    cfg = OmegaConf.load(config_file)
    return cfg

def train(rank, world_size, config_file, outdir):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    
    cfg = _init_config(config_file, outdir)
    
    ## load vqvae model ##
    dvae = DiscreteVAE(**cfg.vqvae)
    dvae_path = cfg.dvae_checkpoint
    load_checkpoint(dvae, dvae_path)
    dvae = dvae.to(rank)
    dvae.eval()
    print(">> vqvae weights restored from:", dvae_path)
    
    model = StableTTS(**cfg["flow_matching_dit"]).to(rank)
    
    model = DDP(model, device_ids=[rank])

    train_dataset = MatchingFlowDataset(cfg, cfg.dataset['training_files'])
    train_sampler = DistributedSampler(train_dataset) if world_size > 1 else None
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=cfg.dataloader.batch_size,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
        drop_last=cfg.dataloader.drop_last,
        pin_memory=cfg.dataloader.pin_memory,
        collate_fn=MatchingFlowCollator()
    )
    
    if rank == 0:
        writer = SummaryWriter(os.path.join(outdir, "summary"))

    optimizer = optim.AdamW(model.parameters(), lr=cfg.train["lr"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=cfg.train["warmup_steps"], num_training_steps=cfg.train["epochs"] * len(train_dataloader))
    
    # load latest checkpoints if possible
    current_epoch = continue_training(os.path.join(outdir, "checkpoint"), model, optimizer)

    model.train()
    for epoch in range(current_epoch, cfg.train["epochs"]):  # loop over the train_dataset multiple times
        train_sampler.set_epoch(epoch)
        if rank == 0:
            dataloader = tqdm(train_dataloader)
        else:
            dataloader = train_dataloader

        for batch_idx, data in enumerate(dataloader):
            if data is None:
                continue

            with torch.no_grad():
                for key in data:
                    data[key] = data[key].to(rank, non_blocking=True)

                    #latent, _ = dvae.infer(data['padded_mel'])
                    latent = dvae.encode_logits(data['padded_mel'])
                    latent = latent.transpose(1, 2)
            #y_start = normalize_tacotron_mel(data['padded_mel'])
            y_start = data['padded_mel']
            y_lengths = data['padded_mel_lengths']
            aligned_conditioning = latent
            #conditioning_latent = normalize_tacotron_mel(data['padded_mel_refer'])
            conditioning_latent = data['padded_mel_refer']
            optimizer.zero_grad()
            loss = model(
                x = latent,
                    y = y_start,
                    y_lengths = y_lengths,
                    z = conditioning_latent,
                    z_lengths = data['padded_mel_refer_lengths']
                )
            loss.backward()
            optimizer.step()
            scheduler.step()

            steps = epoch * len(dataloader) + batch_idx
            if rank == 0 and steps % cfg.train["log_interval"] == 0:
                print([steps, loss.item()])
                grad_norm = get_grad_norm(model.module)
                writer.add_scalar("training/loss", loss.item(), steps)
                writer.add_scalar("training/grad_norm", grad_norm, steps)
                writer.add_scalar("learning_rate/learning_rate", scheduler.get_last_lr()[0], steps)

            if rank == 0 and steps % cfg.train["save_interval"] == 0:
                torch.save(model.module.state_dict(), os.path.join(outdir, "checkpoint", f'checkpoint_step_{steps}.pt'))
                torch.save(optimizer.state_dict(), os.path.join(outdir, "checkpoint", f'optimizer_step_{steps}.pt'))

        torch.save(model.module.state_dict(), os.path.join(outdir, "checkpoint", f'checkpoint_{epoch}.pt'))
        torch.save(optimizer.state_dict(), os.path.join(outdir, "checkpoint", f'optimizer_{epoch}.pt'))
        print(f"Rank {rank}, Epoch {epoch}, Loss {loss.item()}")

    cleanup()

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    config_file = sys.argv[1]
    outdir = sys.argv[2]
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(world_size, config_file, outdir), nprocs=world_size)
