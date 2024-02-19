import os
import random

import torch
import torch.nn.functional as F
import torch.utils.data
from torch import LongTensor
from tqdm import tqdm
import torchaudio

from ttts.gpt.voice_tokenizer import VoiceBpeTokenizer
from ttts.vocoder.feature_extractors import MelSpectrogramFeatures
import json
import os


class DiffusionDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, datafile):
        self.tokenizer = VoiceBpeTokenizer(cfg.dataset['gpt_vocab'])
        self.datalist = []
        with open(datafile, 'r', encoding='utf8') as fin:
            for line in fin:
                self.datalist.append(line.strip())

        self.squeeze = cfg.dataset['squeeze']
        self.sample_rate = cfg.dataset['sample_rate']
        self.mel_extractor = MelSpectrogramFeatures(**cfg.dataset['mel'])

    def __getitem__(self, index):
        # Fetch text and add start/stop tokens.
        line = self.datalist[index]
        # key, wav_path, spkid, language, raw_text, cleand_text
        strs = line.strip().split("|")
        if len(strs) < 6:
            return None
        # [language] + cleand_text
        cleand_text = f"[{strs[3]}] {strs[5]}"
        # print(f"cleand_text: {cleand_text}")
        seqid = self.tokenizer.encode(cleand_text)
        # print(f"seqid: {seqid}")
        text_tokens = LongTensor(seqid)
        # print(f"text_tokens.shape: {text_tokens} {len(text_tokens)}")

        key = strs[0]
        wav_path = strs[1]
        wave, sample_rate = torchaudio.load(wav_path)
        # print(f"wave shape: {wave.shape}, sample_rate: {sample_rate}")
        if wave.size(0) > 1:  # mix to mono
            wave = wave[0].unsqueeze(0)
        if sample_rate != self.sample_rate:
            try:
                transform = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
                wave = transform(wave)
            except Exception as e:
                print(f"Warning: {wav_path}, wave shape: {wave.shape}, sample_rate: {sample_rate}")
                return None

        mel = self.mel_extractor(wave)[0]
        mel_raw = mel
        # print(f"mel_raw.shape: {mel_raw.shape}")

        split = random.randint(int(mel_raw.shape[1]//3), int(mel_raw.shape[1]//3*2))
        if random.random() > 0.5:
            mel_refer = mel_raw[:, split:]
        else:
            mel_refer = mel_raw[:, :split]
        if mel_refer.shape[1] > 300:
            mel_refer = mel_refer[:, :300]

        if mel_raw.shape[1] > 600:
            mel_raw = mel_raw[:, :600]

        return text_tokens, mel_raw, mel_refer

    def __len__(self):
        return len(self.audiopaths_and_text)


class DiffusionCollater():

    def __init__(self):
        pass

    def __call__(self, batch):
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return None
        text_lens = [len(x[0]) for x in batch]
        max_text_len = max(text_lens)

        mel_lens = [x[2].shape[1] for x in batch]
        max_mel_len = max(mel_lens)

        mel_refer_lens = [x[3].shape[1] for x in batch]
        max_mel_refer_len = max(mel_refer_lens)

        texts = []
        mel = []
        mel_refers = []
        # This is the sequential "background" tokens that are used as padding for text tokens, as specified in the DALLE paper.
        for sample in batch:
            text_token, mel_code, mel, mel_refer = sample
            texts.append(F.pad(text_token, (0, max_text_len-len(text_token)), value=0))
            mel.append(F.pad(mel, (0, max_mel_len-mel.shape[1]), value=0))
            mel_refers.append(F.pad(mel_refer, (0, max_mel_refer_len-mel_refer.shape[1]), value=0))

        padded_text = torch.stack(texts)
        padded_mel = torch.stack(mel)
        padded_mel_refer = torch.stack(mel_refers)
        return {
            'padded_text': padded_text,
            'padded_mel': padded_mel,
            'mel_lengths': LongTensor(mel_lens),
            'padded_mel_refer': padded_mel_refer,
            'mel_refer_lengths': LongTensor(mel_refer_lens)
        }


if __name__ == '__main__':
    params = {
        'mode': 'gpt_tts',
        'path': 'E:\\audio\\LJSpeech-1.1\\ljs_audio_text_train_filelist.txt',
        'phase': 'train',
        'n_workers': 0,
        'batch_size': 16,
        'mel_vocab_size': 512,
    }
    cfg = json.load(open('ttts/diffusion/config.json'))
    ds = DiffusionDataset(cfg)
    dl = torch.utils.data.DataLoader(ds, **cfg['dataloader'], collate_fn=DiffusionCollater())
    i = 0
    m = []
    max_text = 0
    max_mel = 0
    for b in tqdm(dl):
        break
