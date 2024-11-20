import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torchaudio
import torchvision
from ttts.utils.utils import load_audio
from tqdm import tqdm
from ttts.vocoder.feature_extractors import MelSpectrogramFeatures, MelSpectrogramFeatures1


class PreprocessedMelDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, audio_paths, is_eval=False):

        self.datalist = []
        with open(audio_paths, 'r', encoding='utf8') as fin:
            for line in fin:
                self.datalist.append(line.strip())

        self.pad_to = cfg['dataset']['pad_to_samples']
        self.squeeze = cfg['dataset']['squeeze']
        self.sample_rate = cfg['dataset']['sample_rate']
        if 'mel_type' in cfg['dataset'] and cfg['dataset']['mel_type'] == "librosa":
            self.mel_extractor = MelSpectrogramFeatures1(**cfg['dataset']['mel'])
        else:
            self.mel_extractor = MelSpectrogramFeatures(**cfg['dataset']['mel'])
        self.is_eval = is_eval

    def __getitem__(self, index):
        try:
            line = self.datalist[index]
            strs = line.strip().split("|")
            wav_path = line if len(strs) == 1 else strs[1]
            wav = load_audio(wav_path, self.sample_rate)
            #print(f"wave shape: {wave.shape}, sample_rate: {sample_rate}")
            if wav is None:
                print(f"Warning: {wav_path} loading error, skip!")
                return None

            mel = self.mel_extractor(wav)
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
        except:
            print(f"Warning: {wav_path} processing error, skip!")
            return None
        return mel

    def __len__(self):
        return len(self.datalist)


class MelCollator:
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch):
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return None
        mels = torch.stack(batch)
        return mels


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
    dl = torch.utils.data.DataLoader(ds, **cfg['dataloader'], collate_fn=MelCollator(cfg))
    i = 0
    for b in dl:
        #pass
        torchvision.utils.save_image((b['mel']+1)/2, f'{i}.png')
        i += 1
        if i > 20:
            break
