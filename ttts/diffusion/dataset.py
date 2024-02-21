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
from ttts.utils.utils import load_audio
import json
import os


class DiffusionDataset(torch.utils.data.Dataset):
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

        mel_raw = self.mel_extractor(wav)[0]
        # print(f"mel_raw.shape: {mel_raw.shape}")

        split = random.randint(int(mel_raw.shape[1]//3), int(mel_raw.shape[1]//3*2))
        if random.random() > 0.5:
            mel_refer = mel_raw[:, split:]
        else:
            mel_refer = mel_raw[:, :split]
        if mel_refer.shape[1] > 200:
            mel_refer = mel_refer[:, :200]

        if mel_raw.shape[1] > 400:
            mel_raw = mel_raw[:, :400]
        wav_length = mel_raw.shape[1] * 256

        return text_tokens, mel_raw, mel_refer, wav_length

    def __len__(self):
        return len(self.datalist)


class DiffusionCollater():

    def __init__(self):
        pass

    def __call__(self, batch):
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return None
        text_lens = [len(x[0]) for x in batch]
        max_text_len = max(text_lens)

        mel_lens = [x[1].shape[1] for x in batch]
        max_mel_len = max(mel_lens)

        mel_refer_lens = [x[2].shape[1] for x in batch]
        max_mel_refer_len = max(mel_refer_lens)

        wav_lens = [x[3] for x in batch]
        max_wav_len = max(wav_lens)

        texts = []
        mels = []
        mel_refers = []
        # This is the sequential "background" tokens that are used as padding for text tokens, as specified in the DALLE paper.
        for sample in batch:
            text_token, mel, mel_refer, wav_len = sample
            texts.append(F.pad(text_token, (0, max_text_len-len(text_token)), value=0))
            mels.append(F.pad(mel, (0, max_mel_len-mel.shape[1]), value=0))
            mel_refers.append(F.pad(mel_refer, (0, max_mel_refer_len-mel_refer.shape[1]), value=0))

        padded_text = torch.stack(texts)
        padded_mel = torch.stack(mels)
        padded_mel_refer = torch.stack(mel_refers)
        return {
            'padded_text': padded_text,
            'text_lengths': LongTensor(text_lens),
            'padded_mel': padded_mel,
            'mel_lengths': LongTensor(mel_lens),
            'padded_mel_refer': padded_mel_refer,
            'mel_refer_lengths': LongTensor(mel_refer_lens),
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
    cfg = json.load(open('ttts/diffusion/config.json'))
    ds = DiffusionDataset(cfg)
    dl = torch.utils.data.DataLoader(ds, **cfg['dataloader'], collate_fn=DiffusionCollater())
    i = 0
    m = []
    max_text = 0
    max_mel = 0
    for b in tqdm(dl):
        break
