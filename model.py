from torch.nn import LSTM, Linear, BatchNorm1d, Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F


class Upsample(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = F.upsample(x, size=(1, x.get_shape().as_list()[2]*2), mode='bilinear')
        x = torch.squeeze(x, 1)
        return x

class Downsample(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[:,::2,:]

class Waveunet(nn.Module):
    def __init__(self,
        n_frames = 16384,
        num_layers = 12,
        filter_size = 15,
        num_filters = 24,
        kernel_size_down = 15,
        kernel_size_up = 5,
        output_filter_size = 1,
        output_activation  = 'tanh',
        strides = 1
    ):

        self.num_layers   = num_layers
        self.enc_conv     = []
        self.dec_conv     = []
        self.skip         = []
        self.dec_num_filt = []
        self.out_frames   = n_frames

        self.leaky = nn.LeakyReLU(alpha=0.2)
        self.bn1   = BatchNorm1d()
        self.us    = Upsample()
        self.ds    = Downsample()
        self.out_conv = self.dec_conv(nn.Conv1D(filters=1,kernel_size=1))

        # Encoding Path
        for layer in range(num_layers):

            num_filters_layer = num_filters + (num_filters * layer)

            self.enc_conv.append(nn.Conv1D(
                in_channels=n_frames,
                out_channels=out_frames,
                filters=num_filters_layer, 
                kernel_size=kernel_size_down,
                padding=(kernel_size_down // 2),
                strides=strides))
            
            self.dec_num_filt.append(num_filters_layer)

            out_frames = out_frames / 2

        self.conv_bottleneck = Conv1D(
            out_channels=out_frames,
            filters=num_filters + (num_filters * num_layers), 
            kernel_size=kernel_size_down,
            padding=(kernel_size_down // 2),
            strides=strides)(x)

        # Decoding Path
        for layer in range(num_layers):

            num_filters_layer = self.dec_num_filt[-layer-1]
            out_frames = out_frames * 2

            self.dec_conv(nn.Conv1D(
                out_channels=out_frames,
                filters=num_filters_layer, 
                kernel_size=kernel_size_up,
                padding=(kernel_size_down // 2),
                strides=strides))


    super(Waveunet, self).__init__()

    def forward(self,x):
        """
        Input: (nb_samples, nb_in_channels)
        Output: (nb_samples, nb_out_channels)
        """

        inputs = x

        # Encoding Path
        for layer in range(self.num_layers):

            num_filters_layer = num_filters + (num_filters * layer)
            out_frames = out_frames / 2

            x = self.enc_conv[layer](x)

            x = self.leaky(x)
            x = self.bn1(x)

            # Save skip connection for decoding path and downsample
            self.skip.append(x)
            x = self.downsample(x)

        x = self.conv_bottleneck(x)

        # Decoding Path
        for layer in range(num_layers):

            skip_layer = self.skip[-layer-1]
            
            # Make sure that num_filter coincide with current layer shape
            num_filter = skip_layer.get_shape().as_list()[2]

            # Upsample and Concatenate
            x = self.us(x)
            x = torch.cat((x, skip_layer),2)
            x = self.dec_conv[layer](x)

            x = self.leaky(x)
            x = self.bn1(x)

        # Last concat is with original input
        x = torch.cat((x, inputs),2)

        # Collapse feature maps into N sources
        y   = torch.tanh(self.out_conv(x))