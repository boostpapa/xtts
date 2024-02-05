import os
import torch
import torchaudio
import argparse
import json
import re
import cv2
import numpy as np

from ttts.vqvae.xtts_dvae import DiscreteVAE
from ttts.vocoder.feature_extractors import MelSpectrogramFeatures, MelSpectrogramFeatures1
from ttts.utils.utils import plot_spectrogram_to_numpy
from ttts.utils.infer_utils import load_model


def get_args():
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--checkpoint', required=True, default="./logs/as/G_8000.pth", help='checkpoint')
    parser.add_argument('--config', required=True, default="./configs/config.json", help='config file')
    parser.add_argument('--outdir', required=True, help='ouput directory')
    parser.add_argument('--test_file', required=True, help='test file')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id for this local rank, -1 for cpu')
    parser.add_argument('--infer', action='store_true', default=False, help='decode vqvae for mel images reconstruction.')
    parser.add_argument('--vqcode', action='store_true', default=False, help='extract vq codes')
    parser.add_argument('--npmel', action='store_true', default=False, help='save reconstrunction mel data.')
    parser.add_argument('--rwav', action='store_true', default=False, help='save reconstrunction wav.')
    args = parser.parse_args()
    return args


def infer(mel, dvae, device):
    mel_img = plot_spectrogram_to_numpy(mel[0, :, :].detach().unsqueeze(-1).cpu())
    mel = mel.to(device).squeeze(1)
    #mel_recon = dvae.infer(mel)[0]
    recon_loss, ssim_loss, commitment_loss, mel_recon = dvae(mel)
    #recon_loss = torch.mean(recon_loss)
    print([recon_loss.item(), ssim_loss.item(), commitment_loss.item()])
    mel_recon_img = plot_spectrogram_to_numpy(mel_recon[0, :, :].detach().unsqueeze(-1).cpu())
    return mel_recon, mel_img, mel_recon_img


def extract_vq(mel, dvae, device):
    mel = mel.to(device).squeeze(1)
    vq_code = dvae.get_codebook_indices(mel)
    return vq_code


def main():
    args = get_args()
    print(args)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    #os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    cfg = json.load(open(args.config))
    dvae = DiscreteVAE(**cfg['vqvae'])
    dvae = dvae.to(device)

    dvae_checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))
    if 'model' in dvae_checkpoint:
        dvae_checkpoint = dvae_checkpoint['model']
    dvae.load_state_dict(dvae_checkpoint, strict=False)

    dvae.eval()

    if args.rwav:
        from vocos import Vocos
        vocos = Vocos.from_pretrained("/speechwork/users/wd007/tts/xtts2/model/charactr/vocos-mel-24khz").to(device)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    sample_rate = cfg['dataset']['sample_rate']
    if 'mel_type' in cfg['dataset'] and cfg['dataset']['mel_type'] == "librosa":
        mel_extractor = MelSpectrogramFeatures1(**cfg['dataset']['mel'])
    else:
        mel_extractor = MelSpectrogramFeatures(**cfg['dataset']['mel'])

    with open(args.test_file) as fin:
        for line in fin:
            wav_path = line.strip()
            print(wav_path)

            wave, sr = torchaudio.load(wav_path)
            # print(f"wave shape: {wave.shape}, sample_rate: {sample_rate}")
            if wave.size(0) > 1:  # mix to mono
                wave = wave[0].unsqueeze(0)
            if sr != sample_rate:
                transform = torchaudio.transforms.Resample(sr, sample_rate)
                wave = transform(wave)
            mel = mel_extractor(wave)

            if args.infer:
                mel_recon, mel_img, mel_recon_img = infer(mel, dvae, device=device)
                fname = re.split(r'/|\.', wav_path)[-2]
                cv2.imwrite(f"{args.outdir}/{fname}.png", mel_img)
                cv2.imwrite(f"{args.outdir}/{fname}_recon.png", mel_recon_img)
                if args.npmel:
                    np.save(f"{args.outdir}/{fname}_recon.npy", mel_recon.detach().cpu().numpy())
                    np.save(f"{args.outdir}/{fname}.npy", mel.detach().cpu().numpy())
                if args.rwav:
                    wav = vocos.decode(mel.to(device))
                    torchaudio.save(f"{args.outdir}/{fname}.wav", wav.detach().cpu(), sample_rate)
                    wav_recon = vocos.decode(mel_recon)
                    torchaudio.save(f"{args.outdir}/{fname}_recon.wav", wav_recon.detach().cpu(), sample_rate)
            elif args.vqcode:
                code = extract_vq(mel, dvae, device=device)
                print(code.tolist())


if __name__ == '__main__':
    main()

