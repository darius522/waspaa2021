from torch.nn import LSTM, Linear, BatchNorm1d, Parameter, Conv1d, ConvTranspose1d
import torch
import torch.nn as nn
import torch.nn.functional as F


class Upsample(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = F.upsample(x, size=(1, x.size()[2]*2), mode='bilinear')
        x = torch.squeeze(x, 1)
        return x

class Downsample(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[:,:,::2]

class Waveunet(nn.Module):

    """
    Input: [W x H x Ch x BatchSize] = [FeatMaps x Num_Spl x Ch x BatchSize]
    Output: (nb_samples, nb_out_channels)
    """

    def __init__(self,
        W = 24,
        H = 16384,
        Ch = 1,
        num_layers = 12,
        filter_size = 15,
        kernel_size_down = 15,
        kernel_size_up = 5,
        output_filter_size = 1,
        output_activation  = 'tanh',
        stride = 1
    ):

        super(Waveunet, self).__init__()

        self.num_layers   = num_layers
        self.enc_conv     = []
        self.dec_conv     = []
        self.skip         = []
        self.dec_num_filt = []
        self.W            = W

        self.leaky = nn.LeakyReLU(negative_slope=0.2)
        #self.bn1   = BatchNorm1d()
        self.us    = Upsample()
        self.ds    = Downsample()

        # Encoding Path
        for layer in range(num_layers):

            out_channels = self.W + (self.W * layer)
            in_channels  = 1 if layer == 0 else out_channels - self.W

            self.enc_conv.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size_down,
                padding=(kernel_size_down // 2),
                stride=stride))
            
            self.dec_num_filt.append(out_channels)

        self.conv_bottleneck = Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size_down,
            padding=(kernel_size_down // 2),
            stride=stride)

        # Decoding Path
        for layer in range(num_layers-1):

            in_channels = self.dec_num_filt[-layer-1]
            out_channels = in_channels - self.W
            print(in_channels)
            print(out_channels)
            self.dec_conv.append(nn.ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size_down,
                padding=(kernel_size_down // 2),
                stride=stride))
        
        self.dec_conv.append(nn.Conv1d(in_channels=self.W,out_channels=1,kernel_size=1))

    def forward(self,x):
        """
        Input: [W x H x Ch x BatchSize] = [FeatMaps x Num_Spl x Ch x BatchSize]
        Output: (nb_samples, nb_out_channels)
        """

        inputs = x

        print("input: "+str(inputs.size()))

        # Encoding Path
        for layer in range(self.num_layers):

            x = self.enc_conv[layer](x)

            x = self.leaky(x)
            #x = self.bn1(x)

            # Save skip connection for decoding path and downsample
            print("enc pre-ds: "+str(x.size()))
            x = self.ds(x)
            self.skip.append(x)
            print("enc post-ds: "+str(x.size()))

        x = self.conv_bottleneck(x)
        print("bottleneck: "+str(x.size()))

        # Decoding Path
        for layer in range(self.num_layers):

            skip_layer = self.skip[-layer-1]
            
            # Make sure that num_filter coincide with current layer shape
            num_filter = skip_layer.size()[2]

            # Upsample and Concatenate
            print("dec pre-us X: "+str(x.size()))
            print("dec pre-us Skip: "+str(skip_layer.size()))
            print(self.dec_conv[layer])
            #x = self.us(x)
            x = torch.cat((x, skip_layer),2)
            print("dec post cat X: "+str(x.size()))
            x = self.dec_conv[layer](x)
            print("dec post-us X: "+str(x.size()))
            print("dec post cat: "+str(x.size()))
            x = self.leaky(x)
            #x = self.bn1(x)