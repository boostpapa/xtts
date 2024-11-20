import logging
import os
import functools
import math
from pathlib import Path
import re
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from io import BytesIO
from pydub import AudioSegment
from ttts.utils.xtransformers import ContinuousTransformerWrapper, RelativePositionBias
import glob
import logging


def get_logger(model_dir, filename="train.log"):
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)
    return logger


def load_audio(audiopath, sampling_rate):
    '''
    if audiopath.endswith(".m4a"):
        sound = AudioSegment.from_file(audiopath)
        buffer = BytesIO()
        sound.export(buffer, format="wav")
        buffer.seek(0)
        audio, sr = torchaudio.load(buffer)
        #audio = np.array(sound.get_array_of_samples(), dtype=np.float32).reshape((sound.channels,-1)) / (1<<(8*sound.sample_width-1))
        #audio = torch.from_numpy(audio)
        #sr = sound.frame_rate
    else:
    '''
    audio, sr = torchaudio.load(audiopath)
    #print(f"wave shape: {audio.shape}, sample_rate: {sr}")

    if audio.size(0) > 1:  # mix to mono
        audio = audio[0].unsqueeze(0)

    if sr != sampling_rate:
        try:
            audio = torchaudio.functional.resample(audio, sr, sampling_rate)
        except Exception as e:
            print(f"Warning: {audiopath}, wave shape: {audio.shape}, sample_rate: {sr}")
            return None
    # clip audio invalid values
    audio.clip_(-1, 1)
    return audio


def get_prompt_slice(audio, max_audio_length=20, min_audio_length=3, sample_rate=24000, is_eval=False):
    max_sample_length = max_audio_length * sample_rate
    min_sample_length = min_audio_length * sample_rate
    rel_clip = audio
    # if eval uses a middle size sample when it is possible to be more reproducible
    if is_eval:
        sample_length = int((min_sample_length + max_sample_length) / 2)
    else:
        sample_length = random.randint(min_sample_length, max_sample_length)
    gap = rel_clip.shape[-1] - sample_length
    if gap < 0 and is_eval:
        sample_length = rel_clip.shape[-1]
    elif gap < 0:
        sample_length = rel_clip.shape[-1] // 2
    gap = rel_clip.shape[-1] - sample_length

    # if eval start always from the position 0 to be more reproducible
    if is_eval:
        rand_start = 0
    else:
        rand_start = random.randint(0, gap)

    rand_end = rand_start + sample_length
    rel_clip = rel_clip[:, rand_start:rand_end]
    return rel_clip


def sin_noise(frequency, amplitude, duration, length):
    # frequency = 440  # 音调频率（Hz）80 1200
    # amplitude = 1.0  # 振幅 0.2
    # duration = 2.0   # 持续时间（秒）
    # fs = 24000       # 采样率

    # 创建电流声波形
    t = np.linspace(0, duration, length, False)
    x = np.sin(2*np.pi*frequency*t) * amplitude

    return x


def musa_noise(power, length, sample_rate=24000):
    noise_power = power**2  # 0.0002 0.001
    time_length = length / sample_rate
    noise_scale = np.sqrt(noise_power * time_length)  # 计算噪声幅值
    noise = np.random.normal(scale=noise_scale, size=length)

    return noise


def augment_audio(audio_data, sample_rate):
    if random.uniform(0, 1) < 0.6:
        return audio_data
    else:
        if random.uniform(0, 1) < 0.3:
            frequency = random.randint(80, 12000)
            amplitude = random.uniform(2, 5) * 0.1
            duration = len(audio_data) / sample_rate
            noise = sin_noise(frequency, amplitude, duration, len(audio_data))

        else:
            power = random.uniform(2, 10) * 0.0001
            noise = musa_noise(power, len(audio_data), sample_rate)

        noisy_audio = audio_data + noise
        return noisy_audio


def tokenize_by_CJK_char(line: str) -> str: 
    """  
    Tokenize a line of text with CJK char.

    Note: All return charaters will be upper case.

    Example:                                                                                                                                                                                                                                                                    
      input = "你好世界是 hello world 的中文"
      output = "你 好 世 界 是 HELLO WORLD 的 中 文"

    Args:
      line:
        The input text.

    Return:
      A new string tokenize by CJK char.
    """
    # The CJK ranges is from https://github.com/alvations/nltk/blob/79eed6ddea0d0a2c212c1060b477fc268fec4d4b/nltk/tokenize/util.py
    pattern = re.compile(
        r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF\uF900-\uFAFF\uFE30-\uFE4F\uFF65-\uFFDC\U00020000-\U0002FFFF])"
    )    
    chars = pattern.split(line.strip().upper())
    return " ".join([w.strip() for w in chars if w.strip()])


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    assert mask.dtype == torch.bool
    assert dtype in [torch.float32, torch.bfloat16, torch.float16]
    mask = mask.to(dtype)
    # attention mask bias
    # NOTE(Mddct): torch.finfo jit issues
    #     chunk_masks = (1.0 - chunk_masks) * torch.finfo(dtype).min
    mask = (1.0 - mask) * -1.0e+10
    return mask


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_paths_with_cache(search_path, cache_path=None):
    out_paths=None
    if cache_path!=None and os.path.exists(cache_path):
        out_paths = torch.load(cache_path)
    else:
        path = Path(search_path)
        out_paths = find_audio_files(path, ['.wav','.m4a','.mp3'])
        if cache_path is not None:
            print("Building cache..")
            torch.save(out_paths, cache_path)
    return out_paths


def find_audio_files(folder_path, suffixes):
    files = []
    for suffix in suffixes:
        files.extend(glob.glob(os.path.join(folder_path, '**', f'*{suffix}'),recursive=True))
    return files


def summarize(writer, global_step, scalars={}, histograms={}, images={}, audios={}, audio_sampling_rate=22050):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats='HWC')
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)


MATPLOTLIB_FLAG = False


def plot_spectrogram_to_numpy(spectrogram):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger('matplotlib')
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10,2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


logger = logging


def clean_checkpoints(path_to_models='logs/44k/', n_ckpts_to_keep=2, sort_by_time=True):
    """Freeing up space by deleting saved ckpts

    Arguments:
    path_to_models    --  Path to the model directory
    n_ckpts_to_keep   --  Number of ckpts to keep, excluding G_0.pth and D_0.pth
    sort_by_time      --  True -> chronologically delete ckpts
                        False -> lexicographically delete ckpts
    """
    ckpts_files = [f for f in os.listdir(path_to_models) if os.path.isfile(os.path.join(path_to_models, f))]
    name_key = (lambda _f: int(re.compile('model-(\d+)\.pt').match(_f).group(1)))
    time_key = (lambda _f: os.path.getmtime(os.path.join(path_to_models, _f)))
    sort_key = time_key if sort_by_time else name_key
    x_sorted = lambda _x: sorted([f for f in ckpts_files if f.startswith(_x) and not f.endswith('_0.pth')], key=sort_key)
    to_del = [os.path.join(path_to_models, fn) for fn in
                (x_sorted('model')[:-n_ckpts_to_keep])]
    del_info = lambda fn: logger.info(f".. Free up space by deleting ckpt {fn}")
    del_routine = lambda x: [os.remove(x), del_info(x)]
    rs = [del_routine(fn) for fn in to_del]


# exponential moving average
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    groups = 32
    if channels <= 16:
        groups = 8
    elif channels <= 64:
        groups = 16
    while channels % groups != 0:
        groups = int(groups / 2)
    assert groups > 2
    return GroupNorm32(groups, channels)


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/output heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, mask=None, rel_pos=None):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        if rel_pos is not None:
            weight = rel_pos(weight.reshape(bs, self.n_heads, weight.shape[-2], weight.shape[-1])).reshape(bs * self.n_heads, weight.shape[-2], weight.shape[-1])
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        if mask is not None:
            # The proper way to do this is to mask before the softmax using -inf, but that doesn't work properly on CPUs.
            mask = mask.repeat(self.n_heads, 1).unsqueeze(1)
            weight = weight * mask
        a = torch.einsum("bts,bcs->bct", weight, v)

        return a.reshape(bs, -1, length)


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        do_checkpoint=True,
        relative_pos_embeddings=False,
    ):
        super().__init__()
        self.channels = channels
        self.do_checkpoint = do_checkpoint
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        # split heads before split qkv
        self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))
        if relative_pos_embeddings:
            self.relative_pos_embeddings = RelativePositionBias(scale=(channels // self.num_heads) ** .5, causal=False, heads=num_heads, num_buckets=32, max_distance=64)
        else:
            self.relative_pos_embeddings = None

    def forward(self, x, mask=None):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv, mask, self.relative_pos_embeddings)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    """

    def __init__(self, channels, use_conv, out_channels=None, factor=4):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.factor = factor
        if use_conv:
            ksize = 5
            pad = 2
            self.conv = nn.Conv1d(self.channels, self.out_channels, ksize, padding=pad)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=self.factor, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    """

    def __init__(self, channels, use_conv, out_channels=None, factor=4, ksize=5, pad=2):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv

        stride = factor
        if use_conv:
            self.op = nn.Conv1d(
                self.channels, self.out_channels, ksize, stride=stride, padding=pad
            )
        else:
            assert self.channels == self.out_channels
            self.op = nn.AvgPool1d(kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(nn.Module):
    def __init__(
            self,
            channels,
            dropout,
            out_channels=None,
            use_conv=False,
            use_scale_shift_norm=False,
            up=False,
            down=False,
            kernel_size=3,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        padding = 1 if kernel_size == 3 else 2

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv1d(channels, self.out_channels, kernel_size, padding=padding),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False)
            self.x_upd = Upsample(channels, False)
        elif down:
            self.h_upd = Downsample(channels, False)
            self.x_upd = Downsample(channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv1d(self.out_channels, self.out_channels, kernel_size, padding=padding)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv1d(
                channels, self.out_channels, kernel_size, padding=padding
            )
        else:
            self.skip_connection = nn.Conv1d(channels, self.out_channels, 1)

    def forward(self, x):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class AudioMiniEncoder(nn.Module):
    def __init__(self,
                 spec_dim,
                 embedding_dim,
                 base_channels=128,
                 depth=2,
                 resnet_blocks=2,
                 attn_blocks=4,
                 num_attn_heads=4,
                 dropout=0,
                 downsample_factor=2,
                 kernel_size=3):
        super().__init__()
        self.init = nn.Sequential(
            nn.Conv1d(spec_dim, base_channels, 3, padding=1)
        )
        ch = base_channels
        res = []
        for l in range(depth):
            for r in range(resnet_blocks):
                res.append(ResBlock(ch, dropout, kernel_size=kernel_size))
            res.append(Downsample(ch, use_conv=True, out_channels=ch*2, factor=downsample_factor))
            ch *= 2
        self.res = nn.Sequential(*res)
        self.final = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            nn.Conv1d(ch, embedding_dim, 1)
        )
        attn = []
        for a in range(attn_blocks):
            attn.append(AttentionBlock(embedding_dim, num_attn_heads,))
        self.attn = nn.Sequential(*attn)
        self.dim = embedding_dim

    def forward(self, x):
        h = self.init(x)
        h = self.res(h)
        h = self.final(h)
        h = self.attn(h)
        return h[:, :, 0]


DEFAULT_MEL_NORM_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/mel_norms.pth')


class TorchMelSpectrogram(nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024, n_mel_channels=80, mel_fmin=0, mel_fmax=8000,
                 sampling_rate=22050, normalize=False, mel_norm_file=DEFAULT_MEL_NORM_FILE):
        super().__init__()
        # These are the default tacotron values for the MEL spectrogram.
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.sampling_rate = sampling_rate
        self.mel_stft = torchaudio.transforms.MelSpectrogram(n_fft=self.filter_length, hop_length=self.hop_length,
                                                             win_length=self.win_length, power=2, normalized=normalize,
                                                             sample_rate=self.sampling_rate, f_min=self.mel_fmin,
                                                             f_max=self.mel_fmax, n_mels=self.n_mel_channels,
                                                             norm="slaney")
        self.mel_norm_file = mel_norm_file
        if self.mel_norm_file is not None:
            self.mel_norms = torch.load(self.mel_norm_file)
        else:
            self.mel_norms = None

    def forward(self, inp):
        if len(inp.shape) == 3:  # Automatically squeeze out the channels dimension if it is present (assuming mono-audio)
            inp = inp.squeeze(1)
        assert len(inp.shape) == 2
        if torch.backends.mps.is_available():
            inp = inp.to('cpu')
        self.mel_stft = self.mel_stft.to(inp.device)
        mel = self.mel_stft(inp)
        # Perform dynamic range compression
        mel = torch.log(torch.clamp(mel, min=1e-5))
        if self.mel_norms is not None:
            self.mel_norms = self.mel_norms.to(mel.device)
            mel = mel / self.mel_norms.unsqueeze(0).unsqueeze(-1)
        return mel


class CheckpointedLayer(nn.Module):
    """
    Wraps a module. When forward() is called, passes kwargs that require_grad through torch.checkpoint() and bypasses
    checkpoint for all other args.
    """
    def __init__(self, wrap):
        super().__init__()
        self.wrap = wrap

    def forward(self, x, *args, **kwargs):
        for k, v in kwargs.items():
            assert not (isinstance(v, torch.Tensor) and v.requires_grad)  # This would screw up checkpointing.
        partial = functools.partial(self.wrap, **kwargs)
        return partial(x, *args)


class CheckpointedXTransformerEncoder(nn.Module):
    """
    Wraps a ContinuousTransformerWrapper and applies CheckpointedLayer to each layer and permutes from channels-mid
    to channels-last that XTransformer expects.
    """
    def __init__(self, needs_permute=True, exit_permute=True, checkpoint=True, **xtransformer_kwargs):
        super().__init__()
        self.transformer = ContinuousTransformerWrapper(**xtransformer_kwargs)
        self.needs_permute = needs_permute
        self.exit_permute = exit_permute

        if not checkpoint:
            return
        for i in range(len(self.transformer.attn_layers.layers)):
            n, b, r = self.transformer.attn_layers.layers[i]
            self.transformer.attn_layers.layers[i] = nn.ModuleList([n, CheckpointedLayer(b), r])

    def forward(self, x, **kwargs):
        if self.needs_permute:
            x = x.permute(0,2,1)
        h = self.transformer(x, **kwargs)
        if self.exit_permute:
            h = h.permute(0,2,1)
        return h
