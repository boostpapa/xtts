from typing import List
import torch
import math

from msf.conv2d_gradfix import conv2d as nvlabs_conv2d, no_weight_gradients

class Discriminators(torch.nn.Module):
    def __init__(
        self,
        stacks,
        channels,
        kernel_size,
        frequency_ranges,
        ) -> None:
        super(Discriminators, self).__init__()

        self.frequency_ranges = frequency_ranges

        self.layer_Dict = torch.nn.ModuleDict()

        for index, frequency_Range in enumerate(frequency_ranges):
            self.layer_Dict['Discriminator_{}'.format(index)] = Discriminator(
                stacks= stacks,
                channels= channels,
                kernel_size= kernel_size,
                frequency_range= frequency_Range
                )

    def forward(
        self,
        x: torch.FloatTensor,
        lengths: torch.LongTensor
        ):
        '''
        x: [Batch, Mel_dim, Time]
        '''
        x = x.transpose(1,2)
        return [
            self.layer_Dict['Discriminator_{}'.format(index)](x, lengths)
            for index in range(len(self.frequency_ranges))
            ]

class Discriminator(torch.nn.Module):
    def __init__(
        self,
        stacks: int,
        kernel_size: int,
        channels: int,
        frequency_range: List[int]
        ) -> None:
        super(Discriminator, self).__init__()

        self.frequency_Range = frequency_range

        self.layer = torch.nn.Sequential()

        previous_Channels = 1
        for index in range(stacks - 1):
            self.layer.add_module('Conv_{}'.format(index), Conv2d(
                in_channels= previous_Channels,
                out_channels= channels,
                kernel_size= kernel_size,
                bias= False,
                w_init_gain= 'linear'
                ))
            self.layer.add_module('Leaky_ReLU_{}'.format(index), torch.nn.LeakyReLU(
                negative_slope= 0.2,
                inplace= True
                ))
            previous_Channels = channels

        self.layer.add_module('Projection', Conv2d(
            in_channels= previous_Channels,
            out_channels= 1,
            kernel_size= 1,
            bias= True,
            w_init_gain= 'linear'
            ))

    def forward(
        self,
        x: torch.FloatTensor,
        lengths: torch.LongTensor
        ):
        '''
        x: [Batch, Mel_dim, Time]
        '''
        sampling_Length = lengths.min()
        mels = []
        for mel, length in zip(x, lengths):
            offset = torch.randint(
                low= 0,
                high= length - sampling_Length + 1,
                size= (1,)
                ).to(x.device)
            mels.append(mel[self.frequency_Range[0]:self.frequency_Range[1], offset:offset + sampling_Length])

        mels = torch.stack(mels).unsqueeze(dim= 1)    # [Batch, 1, Sampled_Dim, Min_Time])

        return self.layer(mels).squeeze(dim= 1) # [Batch, Sampled_Dim, Min_Time]


class Conv2d(torch.nn.Conv2d):
    def __init__(self, w_init_gain= 'relu', clamp: float=None, *args, **kwargs):
        self.w_init_gain = w_init_gain
        self.clamp = clamp

        super().__init__(*args, **kwargs)
        self.runtime_Coef = 1.0 / math.sqrt(self.in_channels * self.kernel_size[0] * self.kernel_size[1])

    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, mean=0.0, std= 1.0)
        if not self.bias is None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor):
        x = nvlabs_conv2d(
            input= x,
            weight= self.weight.to(x.device) * self.runtime_Coef,
            stride= self.stride,
            padding= (int((self.kernel_size[0] - self.stride[0]) / 2), int((self.kernel_size[1] - self.stride[0]) / 2))
            )   # [Batch, Out, Resolution, Resolution]

        if not self.bias is None:
            x += self.bias.to(x.device)[None, :, None, None]

        if not self.clamp is None:
            x.clamp_(-self.clamp, self.clamp)

        return x
