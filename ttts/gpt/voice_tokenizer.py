import re
import click

import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from ttts.gpt.text.symbols import punctuation

# from data.audio.paired_voice_audio_dataset import load_mozilla_cv, load_voxpopuli, load_tsv
# from models.audio.tts.tacotron2 import load_filepaths_and_text
# from models.audio.tts.tacotron2.text.cleaners import english_cleaners


def remove_extraneous_punctuation(word):
    replacement_punctuation = {
        '{': '(', '}': ')',
        '[': '(', ']': ')',
        '`': '\'', '—': '-',
        '—': '-', '`': '\'',
        'ʼ': '\''
    }
    replace = re.compile("|".join([re.escape(k) for k in sorted(replacement_punctuation, key=len, reverse=True)]), flags=re.DOTALL)
    word = replace.sub(lambda x: replacement_punctuation[x.group(0)], word)

    # TODO: some of these are spoken ('@', '%', '+', etc). Integrate them into the cleaners.
    extraneous = re.compile(r'^[@#%_=\$\^&\*\+\\]$')
    word = extraneous.sub('', word)
    return word


class VoiceBpeTokenizer:
    def __init__(self, vocab_file):
        if vocab_file is not None:
            self.tokenizer = Tokenizer.from_file(vocab_file)

    def preprocess_text(self, txt):
        # txt = english_cleaners(txt)
        txt = remove_extraneous_punctuation(txt)
        return txt

    def encode(self, txt):
        txt = self.preprocess_text(txt)
        txt = txt.replace(' ', '[SPACE]')
        return self.tokenizer.encode(txt).ids

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()
        txt = self.tokenizer.decode(seq, skip_special_tokens=False).replace(' ', '')
        txt = txt.replace('[SPACE]', ' ')
        txt = txt.replace('[STOP]', '')
        txt = txt.replace('[UNK]', '')

        return txt


@click.command()
@click.option("--train-file", default="data/all.txt.cleaned")
def train(train_file):
    with open(train_file, 'r', encoding='utf-8') as at:
        ttsd = at.readlines()
    #bcd = datasets.load_dataset('bookcorpus', cache_dir='Z:\\huggingface_datasets\\cache')['train']

    #allowed_characters_re = re.compile(r'^[0-9a-z!@#%_=:;"/, \-\$\^&\*\(\)\+\{\[\]\}\\\.\'\?—–ʼ]+$')
    allowed_characters_re = re.compile(r'^[0-9a-z!:;"/, \-\(\)\.\'\?ʼ，。？：；’‘”“、！…（）]+$')
    def preprocess_word(word, report=False):
        # word = english_cleaners(word)
        word = remove_extraneous_punctuation(word)
        if not bool(allowed_characters_re.match(word)):
            if report and word:
                print(f"REPORTING: '{word}'")
            return ''
        return word
    
    def preprocess_word_new(line, report=False):
        try:
            key, wav, spk, language, text, clean_text = line.strip().split("|")
            ## 保留中英文和指定标点符号
            clean_text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9 " + "".join(punctuation) + r"]+", "", clean_text)
            return clean_text
        except Exception as e:
            print(line)
            return ""

    def batch_iterator(batch_size=1000):
        print("Processing ASR texts.")
        for i in range(0, len(ttsd), batch_size):
            yield [preprocess_word_new(t, True) for t in ttsd[i:i+batch_size]]

        #print("Processing bookcorpus.")
        #for i in range(0, len(bcd), batch_size):
        #    yield [preprocess_word(t) for t in bcd[i:i+batch_size]['text']]

    trainer = BpeTrainer(special_tokens=['[STOP]', '[UNK]', '[SPACE]', '[ZH]', '[EN]', '[JA]'], vocab_size=2048)
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(batch_iterator(), trainer, length=len(ttsd))#+len(bcd))

    print(tokenizer.decode(tokenizer.encode("i was traveling throughhadslfghds the woods in 1235375t137{{}}").ids))

    tokenizer.save('gpt_tts_tokenizer.json')


def test():
    tok = VoiceBpeTokenizer('gpt/gpt_tts_tokenizer.json')
    with open('data/bpe_train-set.txt', 'r', encoding='utf-8') as at:
        ttsd = at.readlines()
        for line in ttsd:
            line = line.strip()
            seq = tok.encode(line)
            out = tok.decode(seq)
            print(f">>>{line}")
            print(f"<<<{out}")

def test_new():
    tokenizer = VoiceBpeTokenizer('/speechwork/users/wd007/tts/xtts2/gpt/s1/data/bpetrian/train/gpt_tts_tokenizer.json')
    text = "qing3 gei2 wo3 bo1 fang4 mad world ."
    seq_ids = tokenizer.encode(text) 
    print(seq_ids)
    out = tokenizer.decode(seq_ids)
    print(out)

if __name__ == '__main__':
    '''
    python script/all_text_to_one_file.py 
    '''
    #train()
    #test()
    test_new()
