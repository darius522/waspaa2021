from torch.nn import BatchNorm1d, Parameter, Conv1d, ConvTranspose1d
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import (get_uniform_distribution, Model)

import modules
     
class Waveunet(nn.Module):

    def __init__(self,
        W = 12,
        H = 16384,
        n_ch = 1,
        num_layers = 7,
        kernel_size_down = 15,
        kernel_size_up = 5,
        output_filter_size = 1,
        stride = 1,
        model = Model.waveunet_skip,
        quant_num_bins = 2**5,
        target_entropy = 1.5
    ):

        super(Waveunet, self).__init__()

        self.num_layers    = num_layers
        self.enc_conv      = nn.ModuleList()
        self.dec_conv      = nn.ModuleList()
        self.bn_enc        = nn.ModuleList()
        self.bn_dec        = nn.ModuleList()
        self.skip_encoders = nn.ModuleList()
        self.skip          = []
        self.dec_num_filt  = []
        self.W             = W
        self.H             = H
        self.channel       = n_ch
        self.kernel_size_down = kernel_size_down
        self.kernel_size_up   = kernel_size_up
        self.stride           = stride
        self.model            = model

        self.leaky = nn.LeakyReLU(negative_slope=0.2)
        self.tanh  = nn.Tanh()
        self.ds    = modules.Downsample()
        self.us    = nn.Upsample(scale_factor=2, mode='linear',align_corners=True)

        # Quant
        self.quant = None
        self.quant_active = False
        self.quant_num_bins = quant_num_bins
        self.quant_alpha = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float32), requires_grad=True)
        self.register_parameter(name='alpha', param=(self.quant_alpha))
        self.quant_bins = torch.nn.Parameter(torch.rand(self.quant_num_bins, requires_grad=True) * (-1.6) + 0.8, requires_grad=True)
        self.register_parameter(name='bins', param=(self.quant_bins))
        # Entropy
        self.target_entropy = target_entropy
        self.entropy_fuzz = 0.45
        self.tau_change   = 0.025

        # Encoding Path
        for layer in range(num_layers):

            self.out_channels = self.W + (self.W * layer)
            self.in_channels  = self.channel if layer == 0 else self.out_channels - self.W

            self.enc_conv.append(nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size_down,
                padding=(self.kernel_size_down // 2),
                stride=self.stride)
                )
            
            self.bn_enc.append(BatchNorm1d(self.out_channels))
            
            self.dec_num_filt.append(self.out_channels)

        # Bottleneck
        self.conv_bottleneck = Conv1d(
            in_channels=self.out_channels,
            out_channels=self.out_channels+self.W,
            kernel_size=self.kernel_size_up,
            padding=(self.kernel_size_up // 2),
            stride=self.stride
            )

        self.bn_enc.append(BatchNorm1d(self.out_channels+self.W))

        bottleneck_shape = (int(self.W * (self.num_layers + 1)), int(self.H / (2 ** self.num_layers)))
        self.quant = modules.ScalarSoftmaxQuantization(
            bins = self.quant_bins,
            alpha = self.quant_alpha,
            code_length = bottleneck_shape[1],
            num_kmean_kernels = self.quant_num_bins,
            feat_maps = bottleneck_shape[0],
            module_name = 'main bottleneck quant'
        )

        # Decoding Path
        for layer in range(num_layers):

            self.out_channels = self.dec_num_filt[-layer-1]
            if self.model == Model.waveunet_no_skip:
                self.in_channels = self.out_channels + self.W
            else:
                self.in_channels = self.dec_num_filt[-layer-1] * 2 + self.W
            
            # If enc skip model, store intermediate AEs
            if self.model == Model.waveunet_enc_skip:
                self.skip_encoders.append(modules.SkipEncoding(
                W=self.dec_num_filt[-layer-1], 
                H=bottleneck_shape[1] * (2 **(layer+1)),
                quant_bins=self.bins,
                quant_alpha=self.alpha,
                module_name='layer skip '+str(layer+1)))

            self.dec_conv.append(nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size_up,
                padding=(self.kernel_size_up // 2),
                stride=self.stride
                ))
            
            self.bn_dec.append(BatchNorm1d(self.out_channels))
        
        # If enc skip model, store input autoencoder
        if self.model == Model.waveunet_enc_skip:
                self.skip_encoders.append(modules.SkipEncoding(W=self.channel, 
                H=self.H,
                quant_bins=self.bins,
                quant_alpha=self.alpha,
                module_name='layer skip '+str(layer+1)))
        
        last_conv_in = self.W if (self.model == Model.waveunet_no_skip) else self.W + self.channel
        self.dec_conv.append(nn.Conv1d(
            in_channels=last_conv_in,
            out_channels=self.channel,
            kernel_size=1,
            padding=0,
            stride=1)
            )

    def forward(self,x):
        """
        Input: [BatchSize x Channel x Samples]
        Output: [BatchSize x Channel x Samples]
        """

        self.skip = []

        # Save original inputs for outputs
        inputs = x

        # Encoding Path
        for layer in range(self.num_layers):
            x = self.enc_conv[layer](x)
            x = self.bn_enc[layer](x)
            x = self.leaky(x)

            # Save skip connection for decoding path and downsample
            if not self.model == Model.waveunet_no_skip:
                self.skip.append(x)

            x = self.ds(x)

        # Bottleneck
        x = self.conv_bottleneck(x)
        x = self.bn_enc[layer+1](x)
        x = self.tanh(x)
    
        if self.quant_active:
            x = self.quant(x)

        # Decoding Path
        for layer in range(self.num_layers):
            # Upsample and Concatenate
            x = self.us(x)

            # If model uses skip connection (either encoded or identity)
            if not self.model == Model.waveunet_no_skip:
                # Identity skip
                if self.model == Model.waveunet_skip:
                    skip_layer = self.skip[-layer-1]
                # Encoded skip
                elif self.model == Model.waveunet_enc_skip:
                    skip_layer = self.skip_encoders[layer](self.skip[-layer-1])

                x = torch.cat((x, skip_layer), 1)

            x = self.dec_conv[layer](x)
            x = self.bn_dec[layer](x)
            x = self.leaky(x)

        # Final concatenation with original input, 1x1 convolution, and tanh output
        if not self.model == Model.waveunet_no_skip:
            if self.model == Model.waveunet_enc_skip:
                inputs = self.skip_encoders[-1](inputs)
            x = torch.cat((x, inputs), 1)
        x = self.dec_conv[-1](x)
        y = self.tanh(x)

        return y

    def entropy_control_update(self):
        '''
        check soft assignment's entropy for each quantizer module.
        adjust quantizer lambda according to target entropy
        '''

        # Get bottleneck quant
        if self.quant.entropies.avg < (self.target_entropy - self.entropy_fuzz):
            self.quant.tau -= self.tau_change
        elif self.quant.entropies.avg > (self.target_entropy + self.entropy_fuzz):
            self.quant.tau += self.tau_change

        if self.model == Model.waveunet_enc_skip:
            for skip in self.skip_encoders:
                if skip.quant.entropies.avg < (self.target_entropy - self.entropy_fuzz):
                    skip.quant.tau -= self.tau_change
                elif skip.quant.entropies.avg > (self.target_entropy + self.entropy_fuzz):
                    self.quant.tau += self.tau_change

    def get_combined_entropy_loss(self):
        '''
        combine all quantizer modules' soft-assignment entropy mean
        '''

        entropies = torch.Tensor([self.quant.entropies.avg])
        taus      = torch.Tensor([self.quant.tau])
        target_entropy = torch.Tensor([self.target_entropy])

        if self.model == Model.waveunet_enc_skip:
            for skip in self.skip_encoders:
                entropies      = torch.cat((entropies, torch.Tensor([skip.quant.entropies.avg])), 0)
                taus           = torch.cat((taus, torch.Tensor([skip.quant.tau])), 0)
                target_entropy = torch.cat((target_entropy, torch.Tensor([self.target_entropy])), 0)

        return (torch.sum(torch.pow((entropies - target_entropy) * taus, 2)) / target_entropy.size()[0])

    def reset_entropy_hists(self):

        self.quant.entropies.reset()
        if self.model == Model.waveunet_enc_skip:
            for skip in self.skip_encoders:
                skip.quant.entropies.reset()

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