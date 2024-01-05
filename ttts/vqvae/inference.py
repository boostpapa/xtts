import os
import torch
import torchaudio
import argparse
import json
import re
import cv2

from ttts.vqvae.xtts_dvae import DiscreteVAE
from ttts.vocoder.feature_extractors import MelSpectrogramFeatures
from ttts.utils.utils import plot_spectrogram_to_numpy


def get_args():
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--checkpoint', required=True, default="./logs/as/G_8000.pth", help='checkpoint')
    parser.add_argument('--config', required=True, default="./configs/config.json", help='config file')
    parser.add_argument('--outdir', required=True, help='ouput directory')
    parser.add_argument('--test_file', required=True, help='test file')
    parser.add_argument('--gpu',
                        type=int,
                        default=0,
                        help='gpu id for this local rank, -1 for cpu')
    args = parser.parse_args()
    return args


def infer(mel_extractor, dvae, wav_path, sr, device):
    wave, sample_rate = torchaudio.load(wav_path)
    # print(f"wave shape: {wave.shape}, sample_rate: {sample_rate}")
    if wave.size(0) > 1:  # mix to mono
        wave = wave[0].unsqueeze(0)
    if sample_rate != sr:
        transform = torchaudio.transforms.Resample(sample_rate, sr)
        wave = transform(wave)

    mel = mel_extractor(wave)
    mel_img = plot_spectrogram_to_numpy(mel[0, :, :].detach().unsqueeze(-1).cpu())
    mel = mel.to(device).squeeze(1)
    #mel_recon = dvae.infer(mel)[0]
    recon_loss, ssim_loss, commitment_loss, mel_recon = dvae(mel)
    #recon_loss = torch.mean(recon_loss)
    print([recon_loss.item(), ssim_loss.item(), commitment_loss.item()])
    mel_recon_img = plot_spectrogram_to_numpy(mel_recon[0, :, :].detach().unsqueeze(-1).cpu())
    return mel_img, mel_recon_img


def main():
    args = get_args()
    print(args)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    cfg = json.load(open(args.config))
    dvae = DiscreteVAE(**cfg['vqvae'])
    dvae = dvae.to(device)

    dvae.eval()
    dvae_checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))
    dvae.load_state_dict(dvae_checkpoint, strict=False)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    sample_rate = cfg['dataset']['sample_rate']
    mel_extractor = MelSpectrogramFeatures(**cfg['dataset']['mel'])
    with open(args.test_file) as fin:
        for line in fin:
            wav_path = line.strip()
            print(wav_path)
            mel_img, mel_recon_img = infer(mel_extractor, dvae, wav_path, sr=sample_rate, device=device)
            img_name = re.split(r'/|\.', wav_path)[-2]
            cv2.imwrite(f"{args.outdir}/{img_name}.png", mel_img)
            cv2.imwrite(f"{args.outdir}/{img_name}_recon.png", mel_recon_img)


if __name__ == '__main__':
    main()

