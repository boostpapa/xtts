import os
import random
import json

import torch
import torch.nn.functional as F
import torch.utils.data
from torch import LongTensor
from tqdm import tqdm
import torchaudio
from collections import defaultdict
import sentencepiece as spm
from ttts.utils.byte_utils import byte_encode
from ttts.utils.utils import tokenize_by_CJK_char

from ttts.gpt.voice_tokenizer import VoiceBpeTokenizer
from ttts.vocoder.feature_extractors import MelSpectrogramFeatures
from ttts.utils.utils import AttrDict, load_audio, get_prompt_slice, get_logger


class GptTTSDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, datafile, is_eval=False):
        if 'gpt_vocab' in cfg.dataset:
            self.tokenizer = VoiceBpeTokenizer(cfg.dataset['gpt_vocab'])
            self.use_spm = False
        else:
            self.tokenizer = spm.SentencePieceProcessor()
            self.tokenizer.load(cfg.dataset['bpe_model'])
            self.use_spm = True
        self.datalist = []
        self.spk2wav = defaultdict(list)
        with open(datafile, 'r', encoding='utf8') as fin:
            for line in fin:
                self.datalist.append(line.strip())
                # key, wav_path, spkid, language, raw_text, cleand_text
                strs = line.strip().split("|")
                self.spk2wav[strs[2]].append(strs[1])

        self.squeeze = cfg.dataset['squeeze']
        self.sample_rate = cfg.dataset['sample_rate']
        self.mel_extractor = MelSpectrogramFeatures(**cfg.dataset['mel'])
        self.is_eval = is_eval

    def __getitem__(self, index):
        try:
            # Fetch text and add start/stop tokens.
            line = self.datalist[index]
            #key, wav_path, spkid, language, raw_text, cleand_text
            # key, wav_path, spkid, language, raw_text
            strs = line.strip().split("|")
            if (self.use_spm and len(strs) < 5) or (not self.use_spm and len(strs) < 6):
                return None

            if not self.use_spm:
                cleand_text = strs[5]
                # [language] + cleand_text
                # cleand_text = f"[{strs[3]}] {cleand_text}"
            else:
                cleand_text = strs[4]
                # cleand_text = f"[{strs[3]}] {cleand_text}"
                cleand_text = byte_encode(tokenize_by_CJK_char(cleand_text.lower()))
            # print(f"cleand_text: {cleand_text}")
            seqid = self.tokenizer.encode(cleand_text)
            #print(f"seqid: {seqid}")
            text = LongTensor(seqid)
            #print(f"text.shape: {text} {len(text)}")

            key = strs[0]
            wav_path = strs[1]
            spkid = strs[2]

            wave = load_audio(wav_path, self.sample_rate)
            if wave is None:
                return None
            mel = self.mel_extractor(wave)[0]
            wav_length = mel.shape[1] * 256
            raw_mel = mel
            #print(f"raw_mel.shape: {raw_mel.shape}")

            cond_wav_path = random.choice(self.spk2wav[spkid])
            cond_wave = load_audio(cond_wav_path, self.sample_rate)
            #cond_wave = wave
            if cond_wave is None:
                return None
            cond_wave_clip = get_prompt_slice(cond_wave, 15, 3, self.sample_rate, self.is_eval)
            cond_mel = self.mel_extractor(cond_wave_clip)[0]
        except:
            return None

        if text.shape[0] > 300 or raw_mel.shape[1] > 2400:
            print(f"Warning: {key} text len {text.shape[0]} exceed 300 or raw mel len {raw_mel.shape[1]} exceed 2400.")
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

        raw_mel_lens = [x[1].shape[1] for x in batch]
        #print(raw_mel_lens)
        max_raw_mel_len = max(raw_mel_lens)
        #print(max_raw_mel_len)
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
            raw_mels.append(F.pad(raw_mel, (0, max_raw_mel_len-raw_mel.shape[1]), value=0))
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
    json_cfg = json.load(open('configs/config.json'))
    cfg = AttrDict(json_cfg)
    ds = GptTTSDataset(cfg, cfg['dataset']['validation_files'])
    dl = torch.utils.data.DataLoader(ds, **cfg['dataloader'], collate_fn=GptTTSCollater(cfg))
    i = 0
    m = []
    max_text = 0
    max_mel = 0
    for batch in tqdm(dl):
        #print(batch['padded_raw_mel'].shape, batch['raw_mel_lengths'])
        break
