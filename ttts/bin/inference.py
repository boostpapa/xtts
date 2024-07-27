import torch
import torchaudio
from omegaconf import OmegaConf
import argparse
import os
import numpy as np

from torch.utils.data import DataLoader
from ttts.vqvae.xtts_dvae import DiscreteVAE
from ttts.gpt.model import UnifiedVoice
from ttts.utils.checkpoint import load_checkpoint

from ttts.diffusion.aa_model import AA_diffusion
from ttts.utils.diffusion import SpacedDiffusion, space_timesteps, get_named_beta_schedule
from ttts.diffusion.aa_model import do_spectrogram_diffusion, normalize_tacotron_mel
from vocos import Vocos
from ttts.bin.dataset import GptTTSDataset, GptTTSCollator


class TTSModel(torch.nn.Module):
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

        self.eval_dataset = GptTTSDataset(self.cfg, args.filelist, is_eval=True)
        self.eval_dataloader = DataLoader(self.eval_dataset, **self.cfg.dataloader_eval, collate_fn=GptTTSCollator(self.cfg))

        self.diffusion, self.gpt, self.dvae, self.vocos, self.eval_dataloader \
            = self.accelerator.prepare(self.diffusion, self.gpt, self.dvae, self.vocos, self.eval_dataloader)

    def infer_batch(self, batch, args):
        with torch.cuda.amp.autocast(enabled=self.dtype is not None, dtype=self.dtype):
            # speech_conditioning_latent, text_inputs, text_lengths, mel_codes, wav_lengths
            input_data = [batch['padded_cond_mel'], batch['padded_text'], batch['text_lengths'],
                          batch['padded_raw_mel'], batch['wav_lens'], batch['cond_mel_lengths']]
            input_data = [d.cuda() for d in input_data]
            keys = batch['keys']

            # get vqvae codes from raw mel
            input_data[3] = self.dvae.get_codebook_indices(input_data[3])

            # latent given vq
            if args.dump_latent or args.dump_diffusion or args.dump_wav:
                latent = self.gpt(*input_data, return_latent=True, clip_inputs=False).transpose(1, 2) #(b, d, s)
                latent_lens = torch.ceil(batch['wav_lens'] / self.gpt.mel_length_compression).long()
                lens = latent_lens
                dump_data = latent
            # mel given latent
            if args.dump_diffusion or args.dump_wav:
                cond_mel = input_data[0]
                #diffusion_conditioning = normalize_tacotron_mel(cond_mel[..., :300])
                diffusion_conditioning = normalize_tacotron_mel(cond_mel)
                upstride = self.gpt.mel_length_compression / 256
                mel = do_spectrogram_diffusion(self.diffusion, self.diffuser, latent, diffusion_conditioning,
                                               upstride, temperature=1.0)
                mel_lens = batch['wav_lens'] / 256
                lens = mel_lens
                dump_data = mel
            if args.dump_wav:
                wav = self.vocos.decode(mel)
                wav = 32767 / max(0.01, torch.max(torch.abs(wav))) * 0.90 * wav
                torch.clip(wav, -32767.0, 32767.0)
                wav_lens = batch['wav_lens']
                lens = wav_lens
                dump_data = wav

            # key, dump data
            dump_datas = []
            i = 0
            for m, len_ in zip(dump_data, lens):
                m = m[..., 0:int(len_)]
                item = [keys[i], m.cpu()]
                dump_datas.append(item)
                i += 1
            return dump_datas

    def infer(self, args):
        for batch_idx, batch in enumerate(self.eval_dataloader):
            with torch.no_grad():
                dump_datas = self.infer_batch(batch, args)
            for item in dump_datas:
                key, data = item
                if not args.dump_wav:
                    print(f"{args.outdir}/{key}.npy")
                    np.save(f"{args.outdir}/{key}.npy", data.numpy())
                else:
                    print(f"{args.outdir}/{key}.wav")
                    torchaudio.save(f"{args.outdir}/{key}.wav", data.unsqueeze(0).type(torch.int16), 24000)


def get_args():
    parser = argparse.ArgumentParser(description='inference model dump feature')
    parser.add_argument('--config', type=str, required=True, help='config file')
    parser.add_argument('--fp16',
                        action='store_true',
                        help='whether to export fp16 model, default false')
    parser.add_argument('--dump_vqvae', action='store_true', help='dump vqvae codes feature')
    parser.add_argument('--dump_latent', action='store_true', help='dump gpt latent feature')
    parser.add_argument('--dump_diffusion', action='store_true', help='dump diffusion mel feature')
    parser.add_argument('--dump_wav', action='store_true', help='dump diffusion wav')
    parser.add_argument('--filelist', type=str, required=True, help='dump filelist')
    parser.add_argument('--outdir', type=str, default="outdir", help='results output dir')
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

    model.infer(args)


if __name__ == '__main__':
    main()



