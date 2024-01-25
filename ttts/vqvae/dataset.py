import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torchaudio
import torchvision
from tqdm import tqdm
from ttts.vocoder.feature_extractors import MelSpectrogramFeatures, MelSpectrogramFeatures1


class PreprocessedMelDataset(torch.utils.data.Dataset):

    def __init__(self, audio_paths, opt):

        self.wav_paths = []
        with open(audio_paths, 'r', encoding='utf8') as fin:
            for line in fin:
                self.wav_paths.append(line.strip())

        self.pad_to = opt['dataset']['pad_to_samples']
        self.squeeze = opt['dataset']['squeeze']
        self.sample_rate = opt['dataset']['sample_rate']
        if 'mel_type' in opt['dataset'] and opt['dataset']['mel_type'] == "librosa":
            self.mel_extractor = MelSpectrogramFeatures1(**opt['dataset']['mel'])
        else:
            self.mel_extractor = MelSpectrogramFeatures(**opt['dataset']['mel'])

    def __getitem__(self, index):

        wav_file = self.wav_paths[index]
        wave, sample_rate = torchaudio.load(wav_file)
        #print(f"wave shape: {wave.shape}, sample_rate: {sample_rate}")
        if wave.size(0) > 1:  # mix to mono
            wave = wave[0].unsqueeze(0)
        if sample_rate != self.sample_rate:
            try:
                transform = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
                wave = transform(wave)
            except:
                print(f"Warning: {wav_file}, wave shape: {wave.shape}, sample_rate: {sample_rate}")
                return None
        #print(f"wave shape: {wave.shape}, sample_rate: {sample_rate}")

        mel = self.mel_extractor(wave)
        #print(f"mel shape: {mel.shape}")

        if mel.shape[-1] >= self.pad_to:
            start = torch.randint(0, mel.shape[-1] - self.pad_to+1, (1,))
            mel = mel[:, :, start:start+self.pad_to]
            mask = torch.zeros_like(mel)
        else:
            mask = torch.zeros_like(mel)
            padding_needed = self.pad_to - mel.shape[-1]
            mel = F.pad(mel, (0, padding_needed))
            mask = F.pad(mask, (0, padding_needed), value=1)
        assert mel.shape[-1] == self.pad_to
        if self.squeeze:
            mel = mel.squeeze()

        return mel

    def __len__(self):
        return len(self.wav_paths)


if __name__ == '__main__':
    params = {
        'mode': 'preprocessed_mel',
        'path': 'Y:\\separated\\large_mel_cheaters',
        'cache_path': 'Y:\\separated\\large_mel_cheaters_win.pth',
        'pad_to_samples': 646,
        'phase': 'train',
        'n_workers': 0,
        'batch_size': 16,
    }
    cfg = json.load(open('configs/config.json'))
    ds = PreprocessedMelDataset(cfg)
    dl = torch.utils.data.DataLoader(ds, **cfg['dataloader'])
    i = 0
    for b in dl:
        #pass
        torchvision.utils.save_image((b['mel']+1)/2, f'{i}.png')
        i += 1
        if i > 20:
            break
