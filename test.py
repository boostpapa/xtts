from pypinyin import lazy_pinyin, Style
import torch
import re
from omegaconf import OmegaConf
from ttts.gpt.model import UnifiedVoice
from ttts.vqvae.xtts_dvae import DiscreteVAE
from ttts.gpt.text.cleaner import clean_text1

MODELS = {
    'vqvae.pth':'/speechwork/users/wd007/tts/xtts2/vqvae/s4/exp/baseline_lossl1_ssim1/epoch_19.pth',
    'gpt.pth': '/speechwork/users/wd007/tts/xtts2/gpt/s2/exp/baseline/epoch_10.pth',
    'clvp2.pth': '',
    'diffusion.pth': '/speechwork/users/wd007/tts/xtts2/diffusion/s2/exp/baseline/epoch_0.pth',
    'vocoder.pth': 'model/pytorch_model.bin',
    'rlg_auto.pth': '',
    'rlg_diffuser.pth': '',
}
device = 'cuda:0'



from ttts.utils.infer_utils import load_model
from ttts.vocoder.feature_extractors import MelSpectrogramFeatures
import torchaudio

config='/speechwork/users/wd007/tts/xtts2/diffusion/s2/configs/config_test.yaml'
cfg = OmegaConf.load(config)

## load gpt model ##
gpt = UnifiedVoice(**cfg.gpt)
gpt_path = cfg.gpt_checkpoint
gpt_checkpoint = torch.load(gpt_path, map_location=device)
gpt_checkpoint = gpt_checkpoint['model'] if 'model' in gpt_checkpoint else gpt_checkpoint
gpt.load_state_dict(gpt_checkpoint, strict=False)
gpt = gpt.to(device)
gpt.eval()
print(">> GPT weights restored from:", gpt_path)
gpt.post_init_gpt2_config(use_deepspeed=False, kv_cache=False, half=False)

## load vqvae model ##
dvae = DiscreteVAE(**cfg.vqvae)
dvae_path = cfg.dvae_checkpoint
dvae_checkpoint = torch.load(dvae_path, map_location=device)
if 'model' in dvae_checkpoint:
    dvae_checkpoint = dvae_checkpoint['model']
dvae.load_state_dict(dvae_checkpoint, strict=False)
dvae = dvae.to(device)
dvae.eval()
print(">> vqvae weights restored from:", dvae_path)

cond_audio = '/cfs/import/tts/opensource/baker_BZNSYP/BZNSYP/Wave_22k/008669.wav'
cond_audio = '/speechwork/users/wd007/tts/fishspeech/academiCodec/s1/test_wav/live_audio2_57.wav'
cond_audio = '/cfs/import/tts/opensource/baker_BZNSYP/BZNSYP/Wave_22k/003261.wav'
cond_audio = '/speechwork/users/wd007/tts/data/bilibili/manual/jiachun/jiachun/speak/ZH/wav/00000001_000019.wav'
cond_audio = '/speechwork/users/wd007/tts/fishspeech/academiCodec/s1/test_wav/chenrui1.wav'
audio,sr = torchaudio.load(cond_audio)
if audio.shape[0]>1:
    audio = audio[0].unsqueeze(0)
audio = torchaudio.transforms.Resample(sr, 24000)(audio)
cond_mel = MelSpectrogramFeatures()(audio).to(device)
print(cond_mel.shape)


auto_conditioning = cond_mel
settings = {'temperature': .8, 'length_penalty': 1.0, 'repetition_penalty': 2.0,
                    'top_p': .8,
                    'cond_free_k': 2.0, 'diffusion_temperature': 1.0}

from ttts.diffusion.train import do_spectrogram_diffusion
from ttts.utils.diffusion import SpacedDiffusion, space_timesteps, get_named_beta_schedule
from ttts.diffusion.aa_model import denormalize_tacotron_mel, normalize_tacotron_mel

from vocos import Vocos
vocos = Vocos.from_pretrained("/speechwork/users/wd007/tts/xtts2/model/charactr/vocos-mel-24khz")

from ttts.gpt.voice_tokenizer import VoiceBpeTokenizer
import torch.nn.functional as F
tokenizer = VoiceBpeTokenizer(cfg.dataset['gpt_vocab'])

diffusion = load_model('diffusion', MODELS['diffusion.pth'], config, device)
diffuser = SpacedDiffusion(use_timesteps=space_timesteps(1000, [50]), model_mean_type='epsilon',
                           model_var_type='learned_range', loss_type='mse', betas=get_named_beta_schedule('linear', 1000),
                           conditioning_free=True, conditioning_free_k=2., sampler='dpm++2m')
diffusion_conditioning = normalize_tacotron_mel(cond_mel)


text = "天空上，火幕蔓延而开，将方圆数以千万计的人类尽数笼罩。而在火幕扩散时，那绚丽火焰之中的人影也是越来越清晰。片刻后，火焰减弱而下，一道黑衫身影，便是清楚的出现在了这片天地之间。"
text = "历史将永远记住同志们的杰出创造和奉献，党和人民感谢你们。"
text = "但我们的损失由谁来补?"
text = "那个等会儿有时间吧那个那个下午三哥要拉个会,跟大家一起对一下下半年规划."
text = "其次是双人下午茶项目，这个项目包含了精美的下午茶套餐, 让您和您的伴侣可以在酒店内享受美食的同时，感受到酒店的温馨和舒适。"

'''
pinyin = ' '.join(lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True))
tokenizer = VoiceBpeTokenizer('ttts/gpt/gpt_tts_tokenizer.json')
text_tokens = torch.IntTensor(tokenizer.encode(pinyin)).unsqueeze(0).to(device)
text_tokens = F.pad(text_tokens, (0, 1))  # This may not be necessary.
text_tokens = text_tokens.to(device)
print(pinyin)
print(text_tokens)
'''

punctuation = ["!", "?", "…", ".", ";", "！", "？", "...", "。", "；"]
pattern = r"(?<=[{0}])\s*".format("".join(punctuation))
sentences = [i for i in re.split(pattern, text) if i.strip() != ""]
print(sentences)

top_p = .8
temperature = .8
autoregressive_batch_size = 1
length_penalty = 1.0
repetition_penalty = 2.0
max_mel_tokens = 600
sampling_rate = 24000
lang = "ZH"
# text_tokens = F.pad(text_tokens,(0,400-text_tokens.shape[1]),value=0)
wavs = []
zero_wav = torch.zeros(1, int(sampling_rate*0.15))
for sent in sentences:
    sent = sent.strip().lower()
    print(sent)
    #pinyin = ' '.join(lazy_pinyin(sent, style=Style.TONE3, neutral_tone_with_five=True))
    norm_text, words = clean_text1(sent, lang)
    cleand_text = f"[{lang}] {' '.join(words)}"
    print(cleand_text)
    text_tokens = torch.IntTensor(tokenizer.encode(cleand_text)).unsqueeze(0).to(device)
    #text_tokens = F.pad(text_tokens, (0, 1))  # This may not be necessary.
    text_tokens = F.pad(text_tokens, (1,0), value=0)
    text_tokens = F.pad(text_tokens, (0,1), value=1)
    text_tokens = text_tokens.to(device)
    print(text_tokens)
    print(text_tokens.shape)
    with torch.no_grad():
        codes = gpt.inference_speech(auto_conditioning, text_tokens,
                                do_sample=True,
                                top_p=top_p,
                                temperature=temperature,
                                num_return_sequences=autoregressive_batch_size,
                                length_penalty=length_penalty,
                                repetition_penalty=repetition_penalty,
                                max_generate_length=max_mel_tokens)
        print(codes)
        mel1, _ = dvae.decode(codes[:, :-1])
        wav1 = vocos.decode(mel1.detach().cpu())
        torchaudio.save('gen1.wav',wav1.detach().cpu(), 24000)

        latent = gpt(auto_conditioning, text_tokens,
                    torch.tensor([text_tokens.shape[-1]], device=text_tokens.device), codes,
                    torch.tensor([codes.shape[-1]*gpt.mel_length_compression], device=text_tokens.device),
                    return_latent=True, clip_inputs=False).transpose(1,2)
        latent.shape

        mel = do_spectrogram_diffusion(diffusion, diffuser, latent, diffusion_conditioning, temperature=1.0).detach().cpu()
        wav = vocos.decode(mel)
        wavs.append(wav)
        #wavs.append(zero_wav)


from IPython.display import Audio
wav = torch.cat(wavs, dim=1)
wav = wav.detach().cpu()
torchaudio.save('gen.wav',wav.detach().cpu(), 24000)
Audio(wav, rate=sampling_rate)
