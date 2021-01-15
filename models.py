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
        n_ch = 1,
        num_layers = 6,
        filter_size = 15,
        kernel_size_down = 15,
        kernel_size_up = 5,
        output_filter_size = 1,
        stride = 1
    ):

        super(Waveunet, self).__init__()

        self.num_layers   = num_layers
        self.enc_conv     = nn.ModuleList()
        self.dec_conv     = nn.ModuleList()
        self.skip         = []
        self.dec_num_filt = []
        self.W            = W
        self.channel      = n_ch
        self.kernel_size_down = kernel_size_down
        self.kernel_size_up   = kernel_size_up
        self.stride           = stride

        self.leaky = nn.LeakyReLU(negative_slope=0.2)
        self.tanh  = nn.Tanh()
        #self.bn1   = BatchNorm1d()
        self.ds    = Downsample()

        # Encoding Path
        for layer in range(num_layers):

            self.out_channels = self.W + (self.W * layer)
            self.in_channels  = self.channel if layer == 0 else self.out_channels - self.W

            self.enc_conv.append(nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size_down,
                padding=(self.kernel_size_down // 2),
                stride=self.stride))
            
            self.dec_num_filt.append(self.out_channels)

        self.conv_bottleneck = Conv1d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size_up,
            padding=(self.kernel_size_up // 2),
            stride=self.stride)

        # Decoding Path
        for layer in range(num_layers-1):

            self.in_channels = self.dec_num_filt[-layer-1]
            self.out_channels = self.in_channels - self.W

            self.dec_conv.append(nn.ConvTranspose1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size_up,
                padding=(self.kernel_size_up // 2),
                stride=self.stride))
        
        self.dec_conv.append(nn.Conv1d(in_channels=self.W,out_channels=self.channel,kernel_size=1,padding=0,stride=1))

    def forward(self,x):
        """
        Input: [W x H x Ch x BatchSize] = [FeatMaps x Num_Spl x Ch x BatchSize]
        Output: (nb_samples, nb_out_channels)
        """

        inputs = x

        # Encoding Path
        for layer in range(self.num_layers):

            x = self.enc_conv[layer](x)

            x = self.leaky(x)
            #x = self.bn1(x)

            # Save skip connection for decoding path and downsample
            x = self.ds(x)
            self.skip.append(x)

        x = self.conv_bottleneck(x)

        # Decoding Path
        for layer in range(self.num_layers):

            skip_layer = self.skip[-layer-1]
            
            # Make sure that num_filter coincide with current layer shape
            num_filter = skip_layer.size()[2]

            # Upsample and Concatenate
            x = torch.cat((x, skip_layer),2)
            x = self.dec_conv[layer](x)

            x = self.leaky(x) if layer < (self.num_layers-1) else self.tanh(x)
            #x = self.bn1(x)

        return x


class U_Net(nn.Module):
    def __init__(self, H, Hc, Hskip, W1, W2):
        super().__init__()
        self.convEnc1 = nn.Conv1d(1, H, W1, stride=W1//2)
        self.convEnc2 = nn.Conv1d(H, H, W2, dilation=2)
        self.convEnc3 = nn.Conv1d(H, H, W2, dilation=4)
        self.convEnc4 = nn.Conv1d(H, Hc, W2, dilation=8)
        
        self.convDec1 = nn.Conv1d(Hc, H, W2, dilation=16)
        self.convDec2 = nn.Conv1d(2*H, H, W2, dilation=32)
        self.convDec3 = nn.Conv1d(2*H, H, W2, dilation=64)
        self.convDec4 = nn.ConvTranspose1d(2*H, 1, W1, stride=W1//2)
        
        self.convSkipEnc1 = nn.Conv1d(H, Hskip, W2)
        self.convSkipDec1 = nn.ConvTranspose1d(Hskip, H, W2)
        self.convSkipEnc2 = nn.Conv1d(H, Hskip, W2)
        self.convSkipDec2 = nn.ConvTranspose1d(Hskip, H, W2)
        self.convSkipEnc3 = nn.Conv1d(H, Hskip, W2)
        self.convSkipDec3 = nn.ConvTranspose1d(Hskip, H, W2)
        

        nn.init.kaiming_normal_(self.convEnc1.weight)
        nn.init.kaiming_normal_(self.convEnc2.weight)
        nn.init.kaiming_normal_(self.convEnc3.weight)
        nn.init.kaiming_normal_(self.convEnc4.weight)
        nn.init.kaiming_normal_(self.convDec1.weight)
        nn.init.kaiming_normal_(self.convDec2.weight)
        nn.init.kaiming_normal_(self.convDec3.weight)

    def forward(self, x):
        h1=torch.relu(self.convEnc1(x))
        h2=torch.relu(self.convEnc2(h1))
        h3=torch.relu(self.convEnc3(h2))
        h4=torch.relu(self.convEnc4(h3))
        h1skip=torch.relu(self.convSkipEnc1(h1))
        h1skip=torch.relu(self.convSkipDec1(h1skip))
        h2skip=torch.relu(self.convSkipEnc2(h2))
        h2skip=torch.relu(self.convSkipDec2(h2skip))
        h3skip=torch.relu(self.convSkipEnc3(h3))
        h3skip=torch.relu(self.convSkipDec3(h3skip))
        h5=torch.relu(self.convDec1(h4))
        h6=torch.relu(self.convDec2(torch.cat((h3skip[...,-h5.size(2):],h5),1)))
        h7=torch.relu(self.convDec3(torch.cat((h2skip[...,-h6.size(2):],h6),1)))
        y=torch.tanh(self.convDec4(torch.cat((h1skip[...,-h7.size(2):],h7),1)))
        return y