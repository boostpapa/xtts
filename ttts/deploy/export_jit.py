
import torch
import torchaudio
import torch.nn.functional as F
from omegaconf import OmegaConf
import argparse
import os

from ttts.vqvae.xtts_dvae import DiscreteVAE
from ttts.gpt.model import UnifiedVoice
from ttts.vocoder.feature_extractors import MelSpectrogramFeatures
from ttts.utils.checkpoint import load_checkpoint

from ttts.diffusion.aa_model import AA_diffusion
from ttts.utils.diffusion import SpacedDiffusion, space_timesteps, get_named_beta_schedule
from ttts.diffusion.aa_model import do_spectrogram_diffusion, normalize_tacotron_mel
from vocos import Vocos

from ttts.gpt.text.cleaner import clean_text1, text_normalize, text_to_sentences
from ttts.gpt.voice_tokenizer import VoiceBpeTokenizer
import sentencepiece as spm
from ttts.utils.byte_utils import byte_encode
from ttts.utils.utils import tokenize_by_CJK_char


class TTSModel(torch.nn.Module):

    def __init__(self, args):
        super().__init__()
        self.cfg = OmegaConf.load(args.config)
        self.dvae = DiscreteVAE(**self.cfg.vqvae)
        dvae_path = self.cfg.dvae_checkpoint
        load_checkpoint(self.dvae, dvae_path)
        self.dvae.eval()
        print(">> vqvae weights restored from:", dvae_path)

        ## load gpt model ##
        self.gpt = UnifiedVoice(**self.cfg.gpt)
        gpt_path = self.cfg.gpt_checkpoint
        load_checkpoint(self.gpt, gpt_path)
        self.gpt.eval()
        print(">> GPT weights restored from:", gpt_path)
        self.gpt.post_init_gpt2_config(use_deepspeed=False, kv_cache=False, half=args.fp16)

        ## load diffusion model ##
        self.diffusion = AA_diffusion(self.cfg)
        diffusion_path = self.cfg.diffusion_checkpoint
        load_checkpoint(self.diffusion, diffusion_path)
        self.diffusion.eval()
        print(">> diffusion weights restored from:", diffusion_path)

        self.vocos = Vocos.from_pretrained(self.cfg.vocoder_model)
        self.diffuser = SpacedDiffusion(use_timesteps=space_timesteps(1000, [10]), model_mean_type='epsilon',
                                   model_var_type='learned_range', loss_type='mse',
                                   betas=get_named_beta_schedule('linear', 1000),
                                   conditioning_free=True, ramp_conditioning_free=False, conditioning_free_k=2., sampler='dpm++2m')

    def forward(self, cond_mel: torch.Tensor,
                text_tokens: torch.IntTensor, text_lens: torch.IntTensor):
        batch_size = text_tokens.shape[0]
        cond_mel_lengths = torch.tensor([cond_mel.shape[-1]]*batch_size, device=text_tokens.device)
        print(cond_mel_lengths)
        print(text_tokens)
        codes = self.gpt.inference_speech(cond_mel,
                                        text_tokens,
                                        cond_mel_lengths=cond_mel_lengths,
                                        do_sample=True,
                                        top_p=.8,
                                        top_k=30,
                                        temperature=0.8,
                                        num_return_sequences=1,
                                        length_penalty=0.0,
                                        num_beams=3,
                                        repetition_penalty=10.0,
                                        max_generate_length=600)
        #codes = codes[:, :-2]
        print(codes)
        print(f"codes shape: {codes.shape}")
        code_lens = []
        for code in codes:
                if self.cfg.gpt.stop_mel_token not in code:
                    code_lens.append(len(code))
                else:
                    #len_ = code.cpu().tolist().index(8193)+1
                    len_ = (code == 8193).nonzero(as_tuple=False)[0]+1
                    code_lens.append(len_-2)
        code_lens = torch.LongTensor(code_lens).cuda()
        print(f"code len: {code_lens}")

        latent = self.gpt(cond_mel,
                        text_tokens,
                        text_lens,
                        codes,
                        code_lens*self.gpt.mel_length_compression,
                        cond_mel_lengths=cond_mel_lengths,
                        return_latent=True, clip_inputs=False).transpose(1, 2)
        print(f"latent shape: {latent.shape}")

        diffusion_conditioning = normalize_tacotron_mel(cond_mel)
        upstride = self.gpt.mel_length_compression / 256
        mel = do_spectrogram_diffusion(self.diffusion, self.diffuser, latent, diffusion_conditioning,
                                       upstride, temperature=1.0)
        print(f"mel shape: {mel.shape}")
        wav = self.vocos.decode(mel)
        print(f"wav shape: {wav.shape}")
        wav = 32767 / max(0.01, torch.max(torch.abs(wav))) * 1.0 * wav
        torch.clip(wav, -32767.0, 32767.0)
        wavs = []
        for w, len_ in zip(wav, code_lens):                                                                                                                                                                                                                         
                w = w[:len_ * self.gpt.mel_length_compression]
                wavs.append(w.unsqueeze(0))
        wav = torch.cat(wavs, dim=1)
        return wav


def get_args():
    parser = argparse.ArgumentParser(description='export your script model')
    parser.add_argument('--config', type=str, required=True, help='config file')
    parser.add_argument('--fp16',
                        action='store_true',
                        help='whether to export fp16 model, default false')
    # args = parser.parse_args()
    args, _ = parser.parse_known_args()

    return args


cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/split2_J5_TTS_女性_愤怒_4.wav'
cond_audio = '/speechwork/users/wd007/tts/xtts2/gpt/s2_v3/bzshort/erbaHappyLow.wav'
text = "是谁给你的胆量这么跟我说话，嗯? 是你的灵主还是你的伙伴？听着，没用的小东西，这里是城下街，不是过家家的学院！停下你无聊至极的喋喋不休，学着用城下街的方式来解决问题！"
lang = "ZH"


def test():
    args = get_args()
    print(args)
    model = TTSModel(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    # Export jit torch script model
    model.eval()
    model.cuda()

    cfg = OmegaConf.load(args.config)
    audio, sr = torchaudio.load(cond_audio)
    if audio.shape[0] > 1:
        audio = audio[0].unsqueeze(0)
    audio = torchaudio.transforms.Resample(sr, 24000)(audio)
    cond_mel = MelSpectrogramFeatures()(audio).cuda()
    print(f"cond_mel shape: {cond_mel.shape}")

    if 'gpt_vocab' in cfg.dataset:
        tokenizer = VoiceBpeTokenizer(cfg.dataset['gpt_vocab'])
        use_spm = False
    else:
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(cfg.dataset['bpe_model'])
        use_spm = True

    sentences = text_to_sentences(text, lang)
    sentences = ['成对或结群活动，食物几乎完全是植物，', '各种水生植物和藻类。具有较强游牧性，', '迁移模式不规律，主要取决于气候条件，', '迁移时会组成成千上万的大团体。它们是所有天鹅中迁徒地最少的物种，', '有时也是居住地筑巢。 当食物稀少']
    print(sentences)

    sens_tokens = []
    for sen in sentences:
        sen = sen.strip().lower()
        print(sen)
        # pinyin = ' '.join(lazy_pinyin(sent, style=Style.TONE3, neutral_tone_with_five=True))
        if not use_spm:
            norm_text, words = clean_text1(sen, lang)
            cleand_text = ' '.join(words)
            # cleand_text = f"[{lang}] {cleand_text}"
        else:
            norm_text = text_normalize(sen, lang)
            cleand_text = norm_text
            cleand_text = tokenize_by_CJK_char(cleand_text)
            # cleand_text = f"[{lang}] {cleand_text}"
            # cleand_text = cleand_text.replace(' ', '[SPACE]')
            print(cleand_text)
            cleand_text = byte_encode(cleand_text)
        print(cleand_text)
        sen_tokens = torch.IntTensor(tokenizer.encode(cleand_text))
        sen_tokens = F.pad(sen_tokens, (1, 0), value=cfg.gpt.start_text_token)
        sen_tokens = F.pad(sen_tokens, (0, 1), value=cfg.gpt.stop_text_token)
        sens_tokens.append(sen_tokens)

    text_lens = [len(x) for x in sens_tokens]
    max_text_len = max(text_lens)
    texts_token = []
    for sen_tokens in sens_tokens:
        sen_tokens = F.pad(sen_tokens, (0, max_text_len - len(sen_tokens)), value=cfg.gpt.stop_text_token)
        texts_token.append(sen_tokens)
    padded_texts = torch.stack(texts_token).cuda()
    text_lens = torch.IntTensor(text_lens)

    cond_mels = cond_mel.repeat(len(sens_tokens), 1, 1)
    wav = model(cond_mels, padded_texts, text_lens).cpu()
    torchaudio.save('gen.wav', wav.type(torch.int16), 24000)


def main():
    args = get_args()
    # No need gpu for model export
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    model = TTSModel(args)
    # Export jit torch script model
    model.eval()


if __name__ == '__main__':
    #main()
    test()

