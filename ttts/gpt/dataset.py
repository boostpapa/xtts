import os
import random
import json

import torch
import torch.nn.functional as F
import torch.utils.data
from torch import LongTensor
from tqdm import tqdm
import torchaudio

from ttts.gpt.voice_tokenizer import VoiceBpeTokenizer
from ttts.vocoder.feature_extractors import MelSpectrogramFeatures


class GptTTSDataset(torch.utils.data.Dataset):
    def __init__(self, datafile, cfg):
        self.tokenizer = VoiceBpeTokenizer(cfg['dataset']['gpt_vocab'])
        self.datalist = []
        with open(datafile, 'r', encoding='utf8') as fin:
            for line in fin:
                self.datalist.append(line.strip())

        self.squeeze = cfg['dataset']['squeeze']
        self.sample_rate = cfg['dataset']['sample_rate']
        self.mel_extractor = MelSpectrogramFeatures(**cfg['dataset']['mel'])

    def __getitem__(self, index):
        try:
            # Fetch text and add start/stop tokens.
            line = self.datalist[index]
            # key, wav, spk, language, text, cleand_text
            strs = line.strip().split("|")
            if len(strs) < 6:
                return None
            cleand_text = strs[5]
            seqid = self.tokenizer.encode(cleand_text)
            text = LongTensor(seqid)

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
            raw_mel = mel

            wav_length = mel.shape[1]*256
            split = random.randint(int(mel.shape[1]//3), int(mel.shape[1]//3*2))
            if random.random() > 0.5:
                cond_mel = mel[:, :split]
            else:
                cond_mel = mel[:, split:]
        except:
            return None

        if text.shape[0] > 400 or mel.shape[1] > 600:
            return None

        return text, raw_mel, cond_mel, wav_length

    def __len__(self):
        return len(self.datalist)


class GptTTSCollater():

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch):
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return None
        text_lens = [len(x[0]) for x in batch]
        max_text_len = max(text_lens)
        # max_text_len = self.cfg['gpt']['max_text_tokens']

        raw_mel_lens = [len(x[1]) for x in batch]
        max_raw_mel_len = max(raw_mel_lens)
        # max_qmel_len = self.cfg['gpt']['max_mel_tokens']

        cond_mel_lens = [x[2].shape[1] for x in batch]
        max_cond_mel_len = max(cond_mel_lens)

        wav_lens = [x[3] for x in batch]
        max_wav_len = max(wav_lens)

        texts = []
        raw_mels = []
        cond_mels = []
        wavs = []
        # This is the sequential "background" tokens that are used as padding for text tokens, as specified in the DALLE paper.
        for sample in batch:
            text, raw_mel, cond_mel, wav = sample
            text = F.pad(text, (0, max_text_len-len(text)), value=0)
            texts.append(text)
            raw_mels.append(F.pad(raw_mel, (0, max_raw_mel_len-len(raw_mel)), value=0))
            cond_mels.append(F.pad(cond_mel, (0, max_cond_mel_len-cond_mel.shape[1]), value=0))

        padded_raw_mel = torch.stack(raw_mels)
        padded_cond_mel = torch.stack(cond_mels)
        padded_texts = torch.stack(texts)
        return {
            'padded_text': padded_texts,
            'text_lengths': LongTensor(text_lens),
            'padded_raw_mel': padded_raw_mel,
            'raw_mel_lengths': LongTensor(raw_mel_lens),
            'padded_cond_mel': padded_cond_mel,
            'cond_mel_lengths': LongTensor(cond_mel_lens),
            'wav_lens': LongTensor(wav_lens)
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
    cfg = json.load(open('configs/config.json'))
    ds = GptTTSDataset(cfg)
    dl = torch.utils.data.DataLoader(ds, **cfg['dataloader'], collate_fn=GptTTSCollater(cfg))
    i = 0
    m = []
    max_text = 0
    max_mel = 0
    for b in tqdm(dl):
        break
