from pypinyin import lazy_pinyin, Style
import torch
import re

MODELS = {
    'vqvae.pth':'model/vae-30.pt',
    'gpt.pth': 'model/gpt-70.pt',
    'clvp2.pth': '',
    'diffusion.pth': 'model/diffusion-855.pt',
    'vocoder.pth': 'model/pytorch_model.bin',
    'rlg_auto.pth': '',
    'rlg_diffuser.pth': '',
}
device = 'cuda:0'



from ttts.utils.infer_utils import load_model
from ttts.vocoder.feature_extractors import MelSpectrogramFeatures
import torchaudio

gpt = load_model('gpt',MODELS['gpt.pth'],'ttts/gpt/config.json',device)
gpt.post_init_gpt2_config(use_deepspeed=False, kv_cache=False, half=False)
cond_audio = '/speechwork/users/wd007/tts/fishspeech/academiCodec/s1/test_wav/live_audio2_57.wav'
cond_audio = '/speechwork/users/wd007/tts/fishspeech/academiCodec/s1/test_wav/chenrui1.wav'
cond_audio = '/cfs/import/tts/opensource/baker_BZNSYP/BZNSYP/Wave_22k/008669.wav'
audio,sr = torchaudio.load(cond_audio)
if audio.shape[0]>1:
    audio = audio[0].unsqueeze(0)
audio = torchaudio.transforms.Resample(sr,24000)(audio)
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
tokenizer = VoiceBpeTokenizer('ttts/gpt/gpt_tts_tokenizer.json')

diffusion = load_model('diffusion',MODELS['diffusion.pth'],'ttts/diffusion/config.yaml',device)
diffuser = SpacedDiffusion(use_timesteps=space_timesteps(1000, [50]), model_mean_type='epsilon',
                           model_var_type='learned_range', loss_type='mse', betas=get_named_beta_schedule('linear', 1000),
                           conditioning_free=True, conditioning_free_k=2., sampler='dpm++2m')
diffusion_conditioning = normalize_tacotron_mel(cond_mel)


text = "其次是双人下午茶项目，这个项目包含了精美的下午茶套餐，让您和您的伴侣可以在酒店内享受美食的同时，感受到酒店的温馨和舒适。"
text = "历史将永远记住同志们的杰出创造和奉献，党和人民感谢你们。"
text = "天空上，火幕蔓延而开，将方圆数以千万计的人类尽数笼罩。而在火幕扩散时，那绚丽火焰之中的人影也是越来越清晰。片刻后，火焰减弱而下，一道黑衫身影，便是清楚的出现在了这片天地之间。"
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
# text_tokens = F.pad(text_tokens,(0,400-text_tokens.shape[1]),value=0)
wavs = []
zero_wav = torch.zeros(size=([sampling_rate*0.15]))
for sent in text:
    sent = sent.strip().lower()
    print(sent)
    pinyin = ' '.join(lazy_pinyin(sent, style=Style.TONE3, neutral_tone_with_five=True))
    text_tokens = torch.IntTensor(tokenizer.encode(pinyin)).unsqueeze(0).to(device)
    text_tokens = F.pad(text_tokens, (0, 1))  # This may not be necessary.
    text_tokens = text_tokens.to(device)
    print(pinyin)
    print(text_tokens)
    print(text_tokens.shape)
    codes = gpt.inference_speech(auto_conditioning, text_tokens,
                                do_sample=True,
                                top_p=top_p,
                                temperature=temperature,
                                num_return_sequences=autoregressive_batch_size,
                                length_penalty=length_penalty,
                                repetition_penalty=repetition_penalty,
                                max_generate_length=max_mel_tokens)
    print(codes)

    latent = gpt(auto_conditioning, text_tokens,
        torch.tensor([text_tokens.shape[-1]], device=text_tokens.device), codes,
        torch.tensor([codes.shape[-1]*gpt.mel_length_compression], device=text_tokens.device),
        return_latent=True, clip_inputs=False).transpose(1,2)
    latent.shape

    mel = do_spectrogram_diffusion(diffusion, diffuser, latent, diffusion_conditioning, temperature=1.0).detach().cpu()
    wav = vocos.decode(mel)
    wavs.append(wav)
    wavs.append(zero_wav)


from IPython.display import Audio
wav = torch.cat(wavs, dim=0)
wav = wav.detach().cpu()
torchaudio.save('gen.wav',wav.detach().cpu(), 24000)
Audio(wav, rate=24000)
