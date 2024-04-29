import os
import random

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
from ttts.utils.utils import load_audio, get_prompt_slice
import json
import os


class DiffusionDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, datafile, is_eval=False):
        self.tokenizer = VoiceBpeTokenizer(cfg.dataset['gpt_vocab'])
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
            # key, wav_path, spkid, language, raw_text, cleand_text
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
                cleand_text = tokenize_by_CJK_char(cleand_text)
                # cleand_text = f"[{strs[3]}] {cleand_text}"
                #cleand_text = cleand_text.replace(' ', '[SPACE]')
                cleand_text = byte_encode(cleand_text)
            # print(f"cleand_text: {cleand_text}")
            seqid = self.tokenizer.encode(cleand_text)
            # print(f"seqid: {seqid}")
            text_tokens = LongTensor(seqid)
            # print(f"text_tokens.shape: {text_tokens} {len(text_tokens)}")

            key = strs[0]
            wav_path = strs[1]
            spkid = strs[2]

            wav = load_audio(wav_path, self.sample_rate)
            if wav is None:
                print(f"Warning: {wav_path} loading error, skip!")
                return None
            mel_raw = self.mel_extractor(wav)[0]
            # print(f"mel_raw.shape: {mel_raw.shape}")

            refer_wav_path = random.choice(self.spk2wav[spkid])
            refer_wav = load_audio(refer_wav_path, self.sample_rate)
            if refer_wav is None:
                print(f"Warning: {wav_path} loading error, skip!")
                return None
            refer_wav_clip = get_prompt_slice(refer_wav, 4, 1, self.sample_rate, self.is_eval)
            mel_refer = self.mel_extractor(refer_wav_clip)[0]

            #mel_refer = get_prompt_slice(mel_raw, 400, 100, 1, self.is_eval)
            if mel_refer.shape[1] > 300:
                mel_refer = mel_refer[:, :300]

            if mel_raw.shape[1] > 400:
                mel_raw = mel_raw[:, :400]
            wav_length = mel_raw.shape[1] * 256
        except:
            print(f"Warning: {wav_path} processing error, skip!")
            return None

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
