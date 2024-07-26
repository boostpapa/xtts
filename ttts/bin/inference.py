import torch
import torchaudio
from omegaconf import OmegaConf
import argparse
import os

from ttts.vqvae.xtts_dvae import DiscreteVAE
from ttts.gpt.model import UnifiedVoice
from ttts.utils.checkpoint import load_checkpoint

from ttts.diffusion.aa_model import AA_diffusion
from ttts.utils.diffusion import SpacedDiffusion, space_timesteps, get_named_beta_schedule
from ttts.diffusion.aa_model import do_spectrogram_diffusion, normalize_tacotron_mel
from vocos import Vocos

from ttts.gpt.voice_tokenizer import VoiceBpeTokenizer
from ttts.gpt.dataset import GptTTSDataset
import sentencepiece as spm
from multiprocessing import Pool


class TTSModel:

    def __init__(self, args):
        super().__init__()
        self.cfg = OmegaConf.load(args.config)

        if args.fp16:
            self.dtype = torch.float16
        else:  # fp32
            self.dtype = None

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
        if args.fp16:
            self.gpt.post_init_gpt2_config(use_deepspeed=True, kv_cache=True, half=args.fp16)
        else:
            self.gpt.post_init_gpt2_config(use_deepspeed=False, kv_cache=False, half=args.fp16)

        ## load diffusion model ##
        self.diffusion = AA_diffusion(self.cfg)
        diffusion_path = self.cfg.diffusion_checkpoint
        load_checkpoint(self.diffusion, diffusion_path)
        self.diffusion.eval()
        print(">> diffusion weights restored from:", diffusion_path)
        self.diffuser = SpacedDiffusion(use_timesteps=space_timesteps(1000, [15]), model_mean_type='epsilon',
                                        model_var_type='learned_range', loss_type='mse',
                                        betas=get_named_beta_schedule('linear', 1000),
                                        conditioning_free=True, ramp_conditioning_free=False, conditioning_free_k=2.,
                                        sampler='dpm++2m')
        if 'vocoder_model' in self.cfg:
            self.vocos = Vocos.from_pretrained(self.cfg.vocoder_model)

    def infer(self):
        eval_dataset = GptTTSDataset(self.cfg, self.cfg.dataset['validation_files'], is_eval=True)
        with torch.cuda.amp.autocast(enabled=self.dtype is not None, dtype=self.dtype):
            for batch_idx, batch in enumerate(self.eval_dataloader):
                # speech_conditioning_latent, text_inputs, text_lengths, mel_codes, wav_lengths
                input_data = [batch['padded_cond_mel'], batch['padded_text'], batch['text_lengths'],
                                batch['padded_raw_mel'], batch['wav_lens']]
                # get vqvae codes from raw mel
                input_data[3] = self.dvae.get_codebook_indices(input_data[3])
                latent = self.gpt(*input_data,
                                  cond_mel_lengths=batch['cond_mel_lengths'],
                                  return_latent=True, clip_inputs=False)

                cond_mel = input_data[0]
                diffusion_conditioning = normalize_tacotron_mel(cond_mel)
                upstride = self.gpt.mel_length_compression / 256
                mel = do_spectrogram_diffusion(self.diffusion, self.diffuser, latent, diffusion_conditioning,
                                               upstride, temperature=1.0)
                mel_lens = batch['wav_lens']/256
                mels = []
                for m, len_ in zip(mel, mel_lens):
                    m = m[..., 0:int(len_)]
                    mels.append(m)
                mel = torch.cat(mels, dim=-1)


def get_args():
    parser = argparse.ArgumentParser(description='inference model dump feature')
    parser.add_argument('--config', type=str, required=True, help='config file')
    parser.add_argument('--fp16',
                        action='store_true',
                        help='whether to export fp16 model, default false')
    parser.add_argument('--dump_vqvae', action='store_true', help='dump vqvae codes feature')
    parser.add_argument('--dump_latent', action='store_true', help='dump gpt latent feature')
    parser.add_argument('--dump_diffusion', action='store_true', help='dump diffusion mel feature')
    parser.add_argument('--filelist', type=str, help='dump filelist')
    parser.add_argument('--outdir', type=str, help='results output dir')
    parser.add_argument('-n', '--njobs', action='store', default=10, type=int, help='#jobs')
    # args = parser.parse_args()
    args, _ = parser.parse_known_args()
    return args


def main():
    args = get_args()
    print(args)

    model = TTSModel(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    model.cuda()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)





if __name__ == '__main__':
    # main()



