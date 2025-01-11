import os
from omegaconf import OmegaConf
import random
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import LongTensor
import sentencepiece as spm
from ttts.utils.byte_utils import byte_encode
from ttts.utils.utils import tokenize_by_CJK_char

from ttts.utils.utils import load_audio, get_prompt_slice
from ttts.utils.utils import augment_audio
from ttts.gpt.voice_tokenizer import VoiceBpeTokenizer
from ttts.vocoder.feature_extractors import MelSpectrogramFeatures


class BigVGANDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, datafile, is_eval=False):
        self.use_spm = False
        self.use_bbpe = False
        self.use_bpe = False
        if 'gpt_vocab' in cfg.dataset:
            self.tokenizer = VoiceBpeTokenizer(cfg.dataset['gpt_vocab'])
        elif 'bbpe_model' in cfg.dataset:
            self.tokenizer = spm.SentencePieceProcessor()
            self.tokenizer.load(cfg.dataset['bbpe_model'])
            self.use_bbpe = True
            self.use_spm = True
        else:
            self.tokenizer = spm.SentencePieceProcessor()
            self.tokenizer.load(cfg.dataset['bpe_model'])
            self.use_bpe = True
            self.use_spm = True
            self.char_ratio = cfg.dataset['char_ratio'] if 'char_ratio' in cfg.dataset else 0.5
            self.pinyin_ratio_sen = cfg.dataset['pinyin_ratio_sen'] if 'pinyin_ratio_sen' in cfg.dataset else 0.2
        print(f"Using spm {self.use_spm}, bbpe {self.use_bbpe}, bpe {self.use_bpe} tokenizer.")

        self.datalist = []
        with open(datafile, 'r', encoding='utf8') as fin:
            for line in fin:
                self.datalist.append(line.strip())

        # self.segment_size = 8192 # 12288 # 24576
        # self.hop_length = 1024
        # self.chunk = self.segment_size // self.hop_length

        self.squeeze = cfg.dataset['squeeze']
        self.sample_rate = cfg.dataset['sample_rate']
        self.max_dur = cfg.dataset['max_dur']*100 if 'max_dur' in cfg.dataset else 2400
        self.mel_extractor = MelSpectrogramFeatures(**cfg.dataset['mel'])
        self.is_eval = is_eval

    def __getitem__(self, index):
        try:
            line = self.datalist[index]
            strs = line.strip().split("|")

            lang = strs[3]
            if not self.use_spm:
                cleand_text = strs[5]
                # [language] + cleand_text
                # cleand_text = f"[{strs[3]}] {cleand_text}"
            else:
                cleand_text = strs[4]
                cleand_text = tokenize_by_CJK_char(cleand_text)
                # cleand_text = f"[{strs[3]}] {cleand_text}"
                #cleand_text = cleand_text.replace(' ', '[SPACE]')
                if self.use_bbpe:
                    cleand_text = byte_encode(cleand_text)
                elif self.use_bpe:
                    if lang == "ZH" and random.random() > self.char_ratio:
                        chars = cleand_text.split()
                        pinyins = tokenize_by_CJK_char(strs[5]).split()
                        if len(chars) == len(pinyins):
                            n = len(chars)
                            num_to_py = int(n * self.pinyin_ratio_sen)
                            indices = random.sample(range(n), num_to_py)
                            for idx in indices:
                                chars[idx] = pinyins[idx]
                            cleand_text = " ".join(chars)

            # print(f"cleand_text: {cleand_text}")
            seqid = self.tokenizer.encode(cleand_text)
            # print(f"seqid: {seqid}")
            text_tokens = LongTensor(seqid)
            # print(f"text_tokens.shape: {text_tokens} {len(text_tokens)}")

            wav_path = strs[1]

            wav = load_audio(wav_path, self.sample_rate)
            if wav is None:
                print(f"Warning: {wav_path} loading error, skip!")
                return None

            wav_len = int(wav.shape[1]/2048)*2048
            end = 4*self.sample_rate if wav_len/2 > 4*self.sample_rate else int(wav_len/2)
            wav_infer = wav[:, :end]
            wav_refer = wav[:, int(wav_len/2):wav_len]

            '''
            audio_data = wav_refer[0].numpy()
            noisy_audio = augment_audio(audio_data, self.sample_rate)
            noisy_audio = torch.FloatTensor(noisy_audio).unsqueeze(0)
            mel_refer = self.mel_extractor(noisy_audio)[0]

            audio_data = wav_infer[0].numpy()
            noisy_audio = augment_audio(audio_data, self.sample_rate)
            noisy_audio = torch.FloatTensor(noisy_audio).unsqueeze(0)
            mel_infer = self.mel_extractor(noisy_audio)[0]
            '''

            mel_refer = self.mel_extractor(wav_refer)[0]
            mel_infer = self.mel_extractor(wav_infer)[0]
            #wav_infer_length = mel_infer.shape[1] * 256

        except:
            print(f"Warning: {wav_path} processing error, skip!")
            return None

        if text_tokens.shape[0] > 400 or mel_refer.shape[1] > self.max_dur/2:
            print(f"Warning: {wav_path} text len {text_tokens.shape[0]} exceed 400 or raw mel len {mel_refer.shape[1]*2} exceed {self.max_dur}.")
            return None

        return text_tokens, mel_refer, mel_infer, wav_infer, wav_refer

    def __len__(self):
        return len(self.datalist)


class BigVGANCollator:
    def __init__(self):
        pass

    def __call__(self, batch):
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return None

        text_lens = [x[0].size(0) for x in batch]
        max_text_len = max(text_lens)

        mel_refer_lens = [x[1].shape[1] for x in batch]
        max_mel_refer_lens = max(mel_refer_lens)

        mel_infer_lens = [x[2].shape[1] for x in batch]
        max_mel_infer_lens = max(mel_infer_lens)

        wav_infer_lens = [x[3].shape[1] for x in batch]
        max_wav_infer_lens = max(wav_infer_lens)

        wav_refer_lens = [x[4].shape[1] for x in batch]
        max_wav_refer_lens = max(wav_refer_lens)

        texts = []
        mel_refers = []
        mel_infers = []
        wav_infers = []
        wav_refers = []
        # This is the sequential "background" tokens that are used as padding for text tokens, as specified in the DALLE paper.
        for sample in batch:
            text_token, mel_refer, mel_infer, wav_infer, wav_refer = sample

            texts.append(F.pad(text_token, (0, max_text_len-len(text_token)), value=0))
            mel_refers.append(F.pad(mel_refer, (0, max_mel_refer_lens-mel_refer.shape[1]), value=0))
            mel_infers.append(F.pad(mel_infer, (0, max_mel_infer_lens-mel_infer.shape[1]), value=0))
            wav_infers.append(F.pad(wav_infer, (0, max_wav_infer_lens-wav_infer.shape[1]), value=0))
            wav_refers.append(F.pad(wav_refer, (0, max_wav_refer_lens-wav_refer.shape[1]), value=0))

        padded_text = torch.stack(texts)
        padded_mel_refer = torch.stack(mel_refers)
        padded_mel_infer = torch.stack(mel_infers)
        padded_wav_infer = torch.stack(wav_infers)
        padded_wav_refer = torch.stack(wav_refers)

        return {
            'padded_text': padded_text,
            'text_lengths': LongTensor(text_lens),
            'padded_mel_refer': padded_mel_refer,
            'mel_refer_lens': LongTensor(mel_refer_lens),
            'padded_mel_infer': padded_mel_infer,
            'mel_infer_lens': LongTensor(mel_infer_lens),
            'padded_wav_infer': padded_wav_infer,
            'wav_infer_lens': LongTensor(wav_infer_lens),
            'padded_wav_refer': padded_wav_refer,
            'wav_refer_lens': LongTensor(wav_refer_lens),
        }
