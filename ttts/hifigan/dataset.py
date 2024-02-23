import os
import random

import torch
import torch.nn.functional as F
import torch.utils.data
from torch import LongTensor
from tqdm import tqdm

from ttts.vocoder.feature_extractors import MelSpectrogramFeatures
from ttts.gpt.voice_tokenizer import VoiceBpeTokenizer
from ttts.utils.utils import load_audio
import json


class HifiGANDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, datafile, is_eval=False):
        self.tokenizer = VoiceBpeTokenizer(cfg.dataset['gpt_vocab'])
        self.datalist = []
        with open(datafile, 'r', encoding='utf8') as fin:
            for line in fin:
                self.datalist.append(line.strip())

        self.squeeze = cfg.dataset['squeeze']
        self.sample_rate = cfg.dataset['sample_rate']
        self.mel_extractor = MelSpectrogramFeatures(**cfg.dataset['mel'])
        self.is_eval = is_eval

    def __getitem__(self, index):
        try:
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

            wav = load_audio(wav_path, self.sample_rate)
            if wav is None:
                return None

            split = random.randint(int(wav.shape[1] // 3), int(wav.shape[1] // 3 * 2))
            if random.random() > 0.5:
                wav_refer = wav[:, split:]
            else:
                wav_refer = wav[:, :split]
            if wav_refer.shape[1] > (200*256):
                wav_refer = wav_refer[:, :200*256]
            mel_refer = self.mel_extractor(wav_refer)[0]

            if wav.shape[1] > 400*256:
                wav = wav[:, :400*256]
            mel = self.mel_extractor(wav)[0]

        except:
            return None

        return text_tokens, mel, wav, mel_refer, wav_refer

    def __len__(self):
        return len(self.datalist)


class HiFiGANCollater():

    def __init__(self):
        pass

    def __call__(self, batch):
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return None
        text_lens = [len(x[0]) for x in batch]
        max_text_len = max(text_lens)

        mel_lens = [len(x[1]) for x in batch]
        max_mel_len = max(mel_lens)

        wav_lens = [x[2].shape[1] for x in batch]
        max_wav_len = max(wav_lens)

        mel_refer_lens = [x[3].shape[1] for x in batch]
        max_mel_refer_len = max(mel_refer_lens)

        wav_refer_lens = [x[4].shape[1] for x in batch]
        max_wav_refer_len = max(wav_refer_lens)

        texts = []
        mels = []
        wavs = []
        mel_refers = []
        wav_refers = []
        for sample in batch:
            text_token, mel, wav, mel_refer, wav_refer = sample
            texts.append(F.pad(text_token, (0, max_text_len-len(text_token)), value=0))
            mels.append(F.pad(mel, (0, max_mel_len-mel.shape[1]), value=0))
            wavs.append(F.pad(wav, (0, max_wav_len-wav.shape[1]), value=0))
            mel_refers.append(F.pad(mel_refer, (0, max_mel_refer_len-mel_refer.shape[1]), value=0))
            wav_refers.append(F.pad(wav_refer, (0, max_wav_refer_len-wav_refer.shape[1]), value=0))

        padded_text = torch.stack(texts)
        padded_mel = torch.stack(mels)
        padded_wav = torch.stack(wavs)
        padded_mel_refer = torch.stack(mel_refers)
        padded_wav_refer = torch.stack(wav_refers)
        return {
            'padded_text': padded_text,
            'text_lengths': LongTensor(text_lens),
            'padded_mel': padded_mel,
            'padded_wav': padded_wav,
            'padded_mel_refer': padded_mel_refer,
            'padded_wav_refer': padded_wav_refer,
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
    cfg = json.load(open('ttts/hifigan/config.json'))
    ds = HifiGANDataset(cfg)
    dl = torch.utils.data.DataLoader(ds, **cfg['dataloader'], collate_fn=HiFiGANCollater())
    i = 0
    m = []
    max_text = 0
    max_mel = 0
    for b in tqdm(dl):
        break