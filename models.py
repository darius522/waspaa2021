from torch.nn import BatchNorm1d, Parameter, Conv1d, ConvTranspose1d
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import (get_uniform_distribution)
import numpy as np

import modules
import utils
     
class Waveunet(nn.Module):

    def __init__(self,
        W = 24,
        H = 16384,
        n_ch = 1,
        num_layers = 8,
        kernel_size_down = 15,
        kernel_size_up = 5,
        output_filter_size = 1,
        stride = 1,
        quant_num_bins = 2**5,
        target_entropy = -1,
        entropy_fuzz = 0.01,
        tau_change = 0.005,
        quant_alpha = -20
    ):

        super(Waveunet, self).__init__()

        self.num_layers    = num_layers
        self.enc_conv      = nn.ModuleList()
        self.dec_conv      = nn.ModuleList()
        self.bn_enc        = nn.ModuleList()
        self.bn_dec        = nn.ModuleList()
        self.skip_encoders = nn.ModuleList()
        self.skip          = []
        self.W             = W
        self.H             = H
        self.channel       = n_ch
        self.kernel_size_down = kernel_size_down
        self.kernel_size_up   = kernel_size_up
        self.stride           = stride
        self.bottleneck_dims  = []

        self.leaky = nn.LeakyReLU(negative_slope=0.2)
        self.tanh  = nn.Tanh()
        self.ds    = modules.Downsample()
        self.us    = nn.Upsample(scale_factor=2, mode='linear',align_corners=True)

        # Quant
        self.quant = None
        self.quant_active = False
        self.quant_num_bins = quant_num_bins
        self.quant_alpha = torch.nn.Parameter(torch.tensor(quant_alpha, dtype=torch.float32), requires_grad=True)
        self.register_parameter(name='alpha', param=(self.quant_alpha))
        self.quant_bins = torch.nn.Parameter(torch.rand(self.quant_num_bins, requires_grad=True) * (-1.6) + 0.8, requires_grad=True)
        self.register_parameter(name='bins', param=(self.quant_bins))
        self.quant_losses = torch.zeros(1,dtype=torch.float)
        # Entropy
        self.target_entropy = -1
        self.entropy_fuzz   = -1
        self.tau_change     = tau_change
        self.code_entropies = torch.zeros(1,2,dtype=torch.float)
        self.quant_losses   = torch.zeros(1,2,dtype=torch.float)
        
        # Encoding Path
        for layer in range(num_layers):

            self.out_channels = self.W
            self.in_channels  = self.W if layer > 0 else 1

            self.enc_conv.append(nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size_down,
                padding=(self.kernel_size_down // 2),
                stride=self.stride)
                )
            
            self.bn_enc.append(BatchNorm1d(self.out_channels))
            
        # Bottleneck
        self.conv_bottleneck = Conv1d(
            in_channels=self.W,
            out_channels=1,
            kernel_size=self.kernel_size_up,
            padding=(self.kernel_size_up // 2),
            stride=self.stride
            )

        self.bn_enc.append(BatchNorm1d(1))

        self.bottleneck_dims = (1, self.H)
        self.quant = modules.ScalarSoftmaxQuantization(
            bins = self.quant_bins,
            alpha = self.quant_alpha,
            code_length = self.bottleneck_dims[1],
            num_kmean_kernels = self.quant_num_bins,
            feat_maps = self.bottleneck_dims[0],
            module_name = 'main bottleneck quant'
        )

        # Decoding Path
        for layer in range(num_layers):

            self.out_channels = self.W
            self.in_channels = 1 if layer == 0 else self.W

            self.dec_conv.append(nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size_down,
                padding=(self.kernel_size_down // 2),
                stride=self.stride)
                )
            
            self.bn_dec.append(BatchNorm1d(self.out_channels))
        
        self.dec_conv.append(nn.Conv1d(
            in_channels=self.W,
            out_channels=self.channel,
            kernel_size=1)
            )

    def forward(self,x):

        self.skip = []
        self.code_entropies = torch.zeros(1,2,dtype=torch.float)
        self.quant_losses   = torch.zeros(1,2,dtype=torch.float)

        # Save original inputs for outputs
        inputs = x

        # Encoding Path
        for layer in range(self.num_layers):
            x = self.enc_conv[layer](x)
            x = self.bn_enc[layer](x)
            x = self.leaky(x)

        # Bottleneck
        x = self.conv_bottleneck(x)
        x = self.bn_enc[layer+1](x)
        x = self.tanh(x)
    
        if self.quant_active:
            x, code_entropy, quant_loss = self.quant.forward_q(x)
            self.code_entropies = code_entropy
            self.quant_losses   = quant_loss

        # Decoding Path
        for layer in range(self.num_layers):

            x = self.dec_conv[layer](x)
            x = self.bn_dec[layer](x)
            x = self.leaky(x)

        x = self.dec_conv[-1](x)
        y = self.tanh(x)

        return y

    def entropy_control_update(self):
        '''
        check soft assignment's entropy for each quantizer module.
        adjust quantizer lambda according to target entropy
        '''
        entropy = self.get_overall_entropy_avg()
        # Get bottleneck quant
        if (entropy < (self.target_entropy - self.entropy_fuzz)):
            self.quant.tau -= self.tau_change
            self.quant.tau2 -= (self.tau_change * 2)
            if (self.quant.tau < 0): self.quant.tau = 0.0
            if (self.quant.tau2 < 0): self.quant.tau2 = 0.0
        elif entropy > (self.target_entropy + self.entropy_fuzz):
            #self.quant.tau += self.tau_change / 2
            # When within acceptable window, make quant loss (alpha) kick in
            self.quant.tau2 += self.tau_change
        
        self.reset_entropy_hists()

    def entropy_loss(self):
        '''
        combine all quantizer modules' soft-assignment entropy mean
        '''
        return torch.sum(self.code_entropies[:,0] * self.code_entropies[:,1])
        #return torch.sum(torch.pow((self.code_entropies[:,0] - self.target_entropy),2) * self.code_entropies[:,1]) / self.code_entropies.size(0)

    def quantization_loss(self):
        '''
        combine all quantizer modules' quantization error. Used the regularize alpha
        '''
        return torch.mean(self.quant_losses[:,0] * self.quant_losses[:,1])
    
    def get_overall_entropy_avg(self):
        avgs = [self.quant.entropy_avg.avg]

        return np.sum(np.asarray(avgs))

    def set_network_entropy_target(self, bitrate, fuzz, sample_rate, frame_size, overlap):
        self.target_entropy = utils.bitrate_to_entropy(bitrate,sample_rate,frame_size,overlap,self.bottleneck_dims)
        self.entropy_fuzz   = utils.bitrate_to_entropy(fuzz,sample_rate,frame_size,overlap,self.bottleneck_dims)

    def reset_entropy_hists(self):
        self.quant.entropy_avg.reset()



class HARPNet(nn.Module):

    def __init__(self,
        W = 24,
        H = 16384,
        n_ch = 1,
        num_layers = 5,
        kernel_size_down = 15,
        kernel_size_up = 5,
        output_filter_size = 1,
        stride = 1,
        quant_num_bins = 2**5,
        target_entropy = -1,
        entropy_fuzz = 0.01,
        num_skips = 1,
        tau_change = 0.0008,
        quant_alpha = -40
    ):

        super(HARPNet, self).__init__()

        self.num_layers    = num_layers
        self.enc_conv      = nn.ModuleList()
        self.dec_conv      = nn.ModuleList()
        self.bn_enc        = nn.ModuleList()
        self.bn_dec        = nn.ModuleList()
        self.skip_encoders = nn.ModuleList()
        self.skip          = []
        self.W             = W
        self.H             = H
        self.channel       = n_ch
        self.kernel_size_down = kernel_size_down
        self.kernel_size_up   = kernel_size_up
        self.stride           = stride
        self.num_skips        = num_skips
        self.bottleneck_dims  = []

        self.leaky = nn.LeakyReLU(negative_slope=0.2)
        self.tanh  = nn.Tanh()
        self.ds    = modules.Downsample()
        self.us    = nn.Upsample(scale_factor=2, mode='linear',align_corners=True)

        # Quant
        self.quant = None
        self.quant_active = False
        self.quant_num_bins = quant_num_bins
        self.quant_alpha = torch.nn.Parameter(torch.tensor(quant_alpha, dtype=torch.float32), requires_grad=True)
        self.register_parameter(name='alpha', param=(self.quant_alpha))
        self.quant_bins = torch.nn.Parameter(torch.rand(self.quant_num_bins, requires_grad=True) * (-1.6) + 0.8, requires_grad=True)
        self.register_parameter(name='bins', param=(self.quant_bins))
        self.quant_losses = torch.zeros(1,dtype=torch.float)
        # Entropy
        self.target_entropy = -1
        self.entropy_fuzz   = -1
        self.tau_change     = tau_change
        self.code_entropies = torch.zeros(1,2,dtype=torch.float)
        self.quant_losses   = torch.zeros(1,2,dtype=torch.float)
        
        # Encoding Path
        for layer in range(num_layers):

            self.out_channels = self.W
            self.in_channels  = self.W if layer > 0 else 1

            self.enc_conv.append(nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size_down,
                padding=(self.kernel_size_down // 2),
                stride=self.stride)
                )
            
            self.bn_enc.append(BatchNorm1d(self.out_channels))
            
        # Bottleneck
        self.conv_bottleneck = Conv1d(
            in_channels=self.W,
            out_channels=1,
            kernel_size=self.kernel_size_up,
            padding=(self.kernel_size_up // 2),
            stride=self.stride
            )

        self.bn_enc.append(BatchNorm1d(1))

        self.bottleneck_dims = (1, self.H)
        self.quant = modules.ScalarSoftmaxQuantization(
            bins = self.quant_bins,
            alpha = self.quant_alpha,
            code_length = self.bottleneck_dims[1],
            num_kmean_kernels = self.quant_num_bins,
            feat_maps = self.bottleneck_dims[0],
            module_name = 'main bottleneck quant'
        )

        # Decoding Path
        for layer in range(num_layers):

            self.out_channels = self.W
            if layer == 0: self.in_channels = 1
            elif layer > 0 and layer <= self.num_skips: self.in_channels = self.W * 2
            else: self.in_channels = self.W
            
            # If enc skip model, store intermediate AEs
            if (layer <= self.num_skips-1):
                self.skip_encoders.append(modules.SkipEncoding(
                W=self.W, 
                W_layer=self.W,
                H=self.H,
                quant_bins=self.quant_bins,
                quant_alpha=self.quant_alpha,
                module_name='layer skip '+str(layer+1)))

            self.dec_conv.append(nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size_down,
                padding=(self.kernel_size_down // 2),
                stride=self.stride)
                )
            
            self.bn_dec.append(BatchNorm1d(self.out_channels))
        
        self.dec_conv.append(nn.Conv1d(
            in_channels=self.W,
            out_channels=self.channel,
            kernel_size=1)
            )

    def forward(self,x):

        self.skip = []
        self.code_entropies = torch.zeros(1,2,dtype=torch.float)
        self.quant_losses   = torch.zeros(1,2,dtype=torch.float)

        # Save original inputs for outputs
        inputs = x

        # Encoding Path
        for layer in range(self.num_layers):
            x = self.enc_conv[layer](x)
            x = self.bn_enc[layer](x)
            x = self.leaky(x)

            # Save skip connection for decoding path
            if (layer >= (self.num_layers - self.num_skips)):
                self.skip.append(x)

        # Bottleneck
        x = self.conv_bottleneck(x)
        x = self.bn_enc[layer+1](x)
        x = self.tanh(x)
    
        if self.quant_active:
            x, code_entropy, quant_loss = self.quant.forward_q(x)
            self.code_entropies = code_entropy
            self.quant_losses   = quant_loss

        # Decoding Path
        for layer in range(self.num_layers):

            x = self.dec_conv[layer](x)
            x = self.bn_dec[layer](x)
            x = self.leaky(x)
            # If model uses skip connection (either encoded or identity)
            if layer <= self.num_skips-1:
                skip_layer, skip_entropy, skip_quant_loss = self.skip_encoders[layer].forward_skip(self.skip[-layer-1])
                if self.quant_active:
                    self.code_entropies = torch.cat((self.code_entropies,skip_entropy))
                    self.quant_losses = torch.cat((self.quant_losses,skip_quant_loss))

                x = torch.cat((x, skip_layer), 1)


        # Final concatenation with original input, 1x1 convolution, and tanh output
        if self.num_skips == self.num_layers+1:
            inputs, input_entropy, input_quant_loss = self.skip_encoders.forward_skip[-1](inputs)
            if self.quant_active:
                self.code_entropies = torch.cat((self.code_entropies,input_entropy))
                self.quant_losses = torch.cat((self.quant_losses,input_quant_loss))
            x = torch.cat((x, inputs), 1)

        x = self.dec_conv[-1](x)
        y = self.tanh(x)

        return y

    def entropy_control_update(self):
        '''
        check soft assignment's entropy for each quantizer module.
        adjust quantizer lambda according to target entropy
        '''
        entropy = self.get_overall_entropy_avg()
        # Get bottleneck quant
        if (entropy < (self.target_entropy - self.entropy_fuzz)):
            self.quant.tau -= self.tau_change
            self.quant.tau2 -= (self.tau_change * 2)
            if (self.quant.tau < 0): self.quant.tau = 0.0
            if (self.quant.tau2 < 0): self.quant.tau2 = 0.0
        elif entropy > (self.target_entropy + self.entropy_fuzz):
            #self.quant.tau += self.tau_change / 2
            # When within acceptable window, make quant loss (alpha) kick in
            self.quant.tau2 += self.tau_change

        for skip in self.skip_encoders:
            # Get skip quant
            if (entropy < (self.target_entropy - self.entropy_fuzz)):
                skip.quant.tau -= self.tau_change
                skip.quant.tau2 -= (self.tau_change * 2)
                if (skip.quant.tau < 0): skip.quant.tau = 0.0
                if (skip.quant.tau2 < 0): skip.quant.tau2 = 0.0
            elif entropy > (self.target_entropy + self.entropy_fuzz):
                #self.quant.tau += self.tau_change / 2
                # When within acceptable window, make quant loss (alpha) kick in
                skip.quant.tau2 += self.tau_change
        
        self.reset_entropy_hists()

    def entropy_loss(self):
        '''
        combine all quantizer modules' soft-assignment entropy mean
        '''
        return torch.sum(self.code_entropies[:,0] * self.code_entropies[:,1])
        #return torch.sum(torch.pow((self.code_entropies[:,0] - self.target_entropy),2) * self.code_entropies[:,1]) / self.code_entropies.size(0)

    def quantization_loss(self):
        '''
        combine all quantizer modules' quantization error. Used the regularize alpha
        '''
        return torch.mean(self.quant_losses[:,0] * self.quant_losses[:,1])
    
    def get_overall_entropy_avg(self):
        avgs = [self.quant.entropy_avg.avg]
        for skip in self.skip_encoders:
            avgs.append(skip.quant.entropy_avg.avg)

        return np.sum(np.asarray(avgs))

    def set_network_entropy_target(self, bitrate, fuzz, sample_rate, frame_size, overlap):
        self.target_entropy = utils.bitrate_to_entropy(bitrate,sample_rate,frame_size,overlap,self.bottleneck_dims)
        self.entropy_fuzz   = utils.bitrate_to_entropy(fuzz,sample_rate,frame_size,overlap,self.bottleneck_dims)
        for l in range(self.num_skips):
            self.target_entropy += utils.bitrate_to_entropy(bitrate,sample_rate,frame_size,overlap,self.skip_encoders[l].bottleneck_dims)
            self.entropy_fuzz   += utils.bitrate_to_entropy(fuzz,sample_rate,frame_size,overlap,self.skip_encoders[l].bottleneck_dims)


    def reset_entropy_hists(self):
        self.quant.entropy_avg.reset()
        for skip in self.skip_encoders:
            skip.quant.entropy_avg.reset()