import torchaudio
import re
import torch
import torch.nn.functional as F

import numpy as np
from omegaconf import OmegaConf
from ttts.flow_matching_dit.model import StableTTS, normalize_tacotron_mel
from ttts.vqvae.xtts_dvae import DiscreteVAE
from ttts.vocoder.feature_extractors import MelSpectrogramFeatures
from ttts.utils.checkpoint import load_checkpoint

from vocos import Vocos

config = 'config.yaml'
device = 'cpu'
flow_matching_path = "exp/base/checkpoint/checkpoint_30.pt"

cfg = OmegaConf.load(config)

## load vqvae model ##
dvae = DiscreteVAE(**cfg.vqvae)
dvae_path = cfg.dvae_checkpoint
load_checkpoint(dvae, dvae_path)
dvae = dvae.to(device)
dvae.eval()
print(">> vqvae weights restored from:", dvae_path)

## load flow matching model ##
flow_matching_model = StableTTS(**cfg["flow_matching_dit"])
#flow_matching_path = cfg.train["checkpoint"]
load_checkpoint(flow_matching_model, flow_matching_path)
flow_matching_model = flow_matching_model.to(device)
flow_matching_model.eval()
print(">> flow matching weights restored from:", flow_matching_path)

#vocos = Vocos.from_pretrained(cfg.vocoder_model)
vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)

cond_audio = 'siyi.wav'
# pypinyin

audio, sr = torchaudio.load(cond_audio)
if audio.shape[0] > 1:
    audio = audio[0].unsqueeze(0)
audio = torchaudio.transforms.Resample(sr, 24000)(audio)
cond_mel = MelSpectrogramFeatures()(audio).to(device)
print(f"cond_mel shape: {cond_mel.shape}")
#flow_matching_conditioning = normalize_tacotron_mel(cond_mel)
flow_matching_conditioning = cond_mel

auto_conditioning = cond_mel

sampling_rate = 24000
wavs = []
wavs1 = []
with torch.no_grad():
    latent = dvae.encode_logits(cond_mel)
    latent = latent.transpose(1, 2)

    mel_output = flow_matching_model.synthesise(latent, torch.tensor([latent.shape[-1]]).to(device), 100, 1.0,
                                               flow_matching_conditioning)
    wav = vocos.decode(mel_output)
    print(latent.shape)
    print(mel_output.shape)
    mel1, _ = dvae.decode_logits(latent.transpose(1, 2))
    wav1 = vocos.decode(mel1)
    # torchaudio.save('gen1.wav',wav1.detach().cpu(), 24000)
    wav1 = 32767 / max(0.01, torch.max(torch.abs(wav1))) * 1.0 * wav1.detach()
    torch.clip(wav1, -32767.0, 32767.0)
    wavs1.append(wav1.detach().cpu())

    wav = 32767 / max(0.01, torch.max(torch.abs(wav))) * 1.0 * wav.detach()
    torch.clip(wav, -32767.0, 32767.0)
    wavs.append(wav.detach().cpu())

wav1 = torch.cat(wavs1, dim=1)
torchaudio.save('gen_dvae.wav', wav1.type(torch.int16), 24000)
#torchaudio.save('gen1.wav', wav1, 24000)

wav = torch.cat(wavs, dim=1)
torchaudio.save('gen.wav', wav.type(torch.int16), 24000)

