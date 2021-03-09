from torch.nn import BatchNorm1d, Parameter, Conv1d, ConvTranspose1d
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

from random import randrange

class SubPixelResolution(nn.Module):
    def __init__(self, in_channels, out_channels,upscaler=2):
        super().__init__()
        self.upscaler = upscaler
        self.conv1d = Conv1d(in_channels=in_channels, out_channels=out_channels*4, kernel_size=1)
        self.ps = nn.PixelShuffle(self.upscaler)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv1d(x)
        x = torch.unsqueeze(x,-1)
        x = self.ps(x)
        
        return x[:,:,:,0]

class Downsample(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[:,:,::2]

class SkipEncoding(nn.Module):
    def __init__(self,         
        W_layer = 24, # maps per layer
        W = 24, # input maps
        H = 16384, # input samples
        num_layers = 3 ,
        kernel_size_down = 15,
        kernel_size_up = 5,
        stride = 1,
        quant_bins = None,
        quant_alpha = -10.0,
        module_name=''
    ):
        super(SkipEncoding, self).__init__()

        self.num_layers   = num_layers
        self.enc_conv     = nn.ModuleList()
        self.dec_conv     = nn.ModuleList()
        self.bn_enc       = nn.ModuleList()
        self.bn_dec       = nn.ModuleList()
        self.dec_num_filt = []
        self.W            = W
        self.W_layer      = W_layer
        self.H            = H
        self.kernel_size_down = kernel_size_down
        self.kernel_size_up   = kernel_size_up
        self.stride           = stride
        self.module_name = module_name

        self.leaky = nn.LeakyReLU(negative_slope=0.2)
        self.ds    = Downsample()
        self.us    = nn.Upsample(scale_factor=2, mode='linear',align_corners=True)
        self.bottleneck_dims = []

        # Quant
        self.quant = None
        self.quant_active = False
        self.quant_bins = quant_bins
        self.quant_alpha = quant_alpha
        print('skip, bins:',self.quant_bins.data_ptr)
        print('skip, alpha:',self.quant_alpha.data_ptr)

        # Encoding Path
        for layer in range(num_layers):

            self.out_channels = self.W_layer
            self.in_channels  = self.W_layer if layer > 0 else self.W

            self.enc_conv.append(nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size_down,
                padding=(self.kernel_size_down // 2),
                stride=self.stride))
            
            self.bn_enc.append(BatchNorm1d(self.out_channels))
            
            self.dec_num_filt.append(self.out_channels)

        # Bottleneck
        self.conv_bottleneck = Conv1d(
            in_channels=self.W_layer,
            out_channels=1,
            kernel_size=self.kernel_size_up,
            padding=(self.kernel_size_up // 2),
            stride=self.stride
            )

        self.bn_enc.append(BatchNorm1d(1))

        self.bottleneck_dims = (1, self.H)
        self.quant = ScalarSoftmaxQuantization(
            bins = self.quant_bins,
            alpha = self.quant_alpha,
            code_length = self.bottleneck_dims[1],
            num_kmean_kernels = self.quant_bins.shape[0],
            feat_maps = self.bottleneck_dims[0],
            module_name = module_name
        )

        # Decoding Path
        for layer in range(num_layers):

            self.out_channels = self.W_layer
            self.in_channels  = self.W_layer if layer > 0 else 1

            self.dec_conv.append(nn.ConvTranspose1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size_up,
                padding=(self.kernel_size_up // 2),
                stride=self.stride))
            
            self.bn_dec.append(BatchNorm1d(self.out_channels))

    def forward_skip(self, x):

        weighted_code_entropy = torch.zeros(1,2,dtype=torch.float)
        quant_loss = torch.zeros(1,dtype=torch.float)

        # Encoding Path
        for layer in range(self.num_layers):

            x = self.enc_conv[layer](x)
            x = self.bn_enc[layer](x)
            x = self.leaky(x)

        # # Bottleneck
        x = self.conv_bottleneck(x)
        x = self.bn_enc[-1](x)
        if self.quant_active:
            x, code_entropy, quant_loss = self.quant.forward_q(x)
            weighted_code_entropy = code_entropy
        x = self.leaky(x)

        # Decoding Path
        for layer in range(self.num_layers):

            x = self.dec_conv[layer](x)
            x = self.bn_dec[layer](x)
            x = self.leaky(x)

        return x, weighted_code_entropy, quant_loss

class ScalarSoftmaxQuantization(nn.Module):
    def __init__(self, 
        bins,        
        alpha,
        code_length,
        num_kmean_kernels,
        feat_maps,
        module_name=''
        ):
        
        super(ScalarSoftmaxQuantization, self).__init__()

        self.bins = bins
        self.alpha = alpha
        self.code_length = code_length
        self.feat_maps   = feat_maps
        self.num_kmean_kernels = num_kmean_kernels

        # Entropy control
        self.code_entropy = 0
        self.tau  = 0
        self.tau2 = 0
        self.entropy_avg = utils.AverageMeter()
    
    def forward_q(self, x):

        '''
        x = [batch_size, feature_maps, floating_code] // [-1, 21, 256]
        bins = [quant_num_bins] // [4]
        output = [-1, 21, 256]
        '''

        input_size = x.size()
        weighted_code_entropy = torch.zeros(1,2,dtype=torch.float)
        weighted_quant_loss   = torch.zeros(1,2,dtype=torch.float)

        x = torch.unsqueeze(x, len(x.size()))
        floating_code = x.expand(input_size[0],self.feat_maps,self.code_length,self.num_kmean_kernels)

        bins_expand = torch.reshape(self.bins, (1, 1, 1, -1))
        dist = torch.abs(floating_code - bins_expand) # [-1, 21, 256, 4]
        soft_assignment = nn.Softmax(len(dist.size())-1)(torch.mul(self.alpha, dist))

        max_prob_bin = torch.topk(soft_assignment,1).indices

        if not self.training:
            hard_assignment = torch.reshape(F.one_hot(max_prob_bin, self.num_kmean_kernels),
                                        (input_size[0], self.feat_maps, self.code_length, self.num_kmean_kernels))
            hard_assignment = hard_assignment.type(torch.float)
            assignment = hard_assignment
        else:
            assignment = soft_assignment + 1e-10

            # Quantization regularization term, prevent alpha from getting to big
            weighted_quant_loss[:,0] = torch.sum(torch.mean(torch.sqrt(assignment),(0,1,2)))
            weighted_quant_loss[:,1] = self.tau2

            p = torch.mean(assignment,dim=(0,1,2))
            # q = torch.histc(floating_code,bins=self.num_kmean_kernels)
            # q /= torch.sum(q)
            # Compute entropy loss
            self.code_entropy = -torch.sum(torch.mul(p,torch.log(p)))
            self.entropy_avg.update(self.code_entropy.detach())

            # Weighted entropy regularization term
            weighted_code_entropy[:,0] = self.code_entropy
            weighted_code_entropy[:,1] = self.tau
                
        bit_code = torch.matmul(assignment,self.bins)
        
        return bit_code, weighted_code_entropy, weighted_quant_loss