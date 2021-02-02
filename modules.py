from torch.nn import BatchNorm1d, Parameter, Conv1d, ConvTranspose1d
import torch
import torch.nn as nn
import torch.nn.functional as F

class Downsample(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[:,:,::2]

class SkipEncoding(nn.Module):
    def __init__(self,         
        W_layer = 12, # maps per layer
        W = 24, # input maps
        H = 16384, # input samples
        num_layers = 3,
        kernel_size_down = 15,
        kernel_size_up = 5,
        stride = 1
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
        self.kernel_size_down = kernel_size_down
        self.kernel_size_up   = kernel_size_up
        self.stride           = stride

        self.leaky = nn.LeakyReLU(negative_slope=0.2)
        self.ds    = Downsample()
        self.us    = nn.Upsample(scale_factor=2, mode='linear',align_corners=True)

        # Encoding Path
        for layer in range(num_layers):

            self.out_channels = self.W + (self.W_layer * (layer+1))
            self.in_channels  = self.out_channels - self.W_layer

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
            in_channels=self.out_channels,
            out_channels=self.out_channels+self.W_layer,
            kernel_size=self.kernel_size_up,
            padding=(self.kernel_size_up // 2),
            stride=self.stride)

        self.bn_enc.append(BatchNorm1d(self.out_channels+self.W_layer))

        # Decoding Path
        for layer in range(num_layers):

            self.out_channels = self.dec_num_filt[-layer-1] - self.W_layer if (layer == num_layers-1) else self.dec_num_filt[-layer-1]
            self.in_channels = self.dec_num_filt[-layer-1] + self.W_layer

            self.dec_conv.append(nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size_up,
                padding=(self.kernel_size_up // 2),
                stride=self.stride))
            
            self.bn_dec.append(BatchNorm1d(self.out_channels))

    def forward(self, x):

        # Encoding Path
        for layer in range(self.num_layers):

            x = self.enc_conv[layer](x)
            x = self.bn_enc[layer](x)
            x = self.leaky(x)
            x = self.ds(x)

        # Bottleneck
        x = self.conv_bottleneck(x)
        x = self.bn_enc[-1](x)

        # Decoding Path
        for layer in range(self.num_layers):

            # Upsample and Concatenate
            x = self.us(x)

            x = self.dec_conv[layer](x)
            x = self.bn_dec[layer](x)
            x = self.leaky(x)

        return x 

class ScalarSoftmaxQuantization(nn.Module):
    def __init__(self,         
        alpha,
        bins, # Quantization bins
        is_quan_on, # Wether quant is on or not
        the_share, # Wether training or testing
        code_length,
        num_kmean_kernels,
        feat_maps
    ):
        super(ScalarSoftmaxQuantization, self).__init__()
    
        self.alpha = -50 #tbd
        self.bins = bins
        self.is_quan_on = is_quan_on
        self.the_share = the_share
        self.code_length = code_length
        self.feat_maps   = feat_maps
        self.num_kmean_kernels = num_kmean_kernels
    
    def forward(self, x):

        '''
        x = [batch_size, feature_maps, floating_code] // [-1, 21, 256]
        bins = [quant_num_bins] // [4]

        output = [-1, 21, 256]
        '''

        input_size = x.size()

        x = torch.unsqueeze(x, len(x.size()))
        floating_code = x.expand(input_size[0],self.feat_maps,self.code_length,self.num_kmean_kernels)

        bins_expand = torch.reshape(self.bins, (1, 1, 1, -1))
        dist = torch.abs(floating_code - bins_expand) # [-1, 21, 256, 4]
        # print("Input: "+str(floating_code[0,0,200,:]))
        # print("Bins: "+str(bins_expand[0,0,0,:]))
        # print("Dist: "+str(dist[0,0,200,:]))
        bottle_neck_size = floating_code.size()[1]
        soft_assignment = nn.Softmax(len(dist.size())-1)(torch.mul(self.alpha, dist))
        soft_assignment_3d = soft_assignment
        # print("SA:"+str(soft_assignment[0,0,200,:]))
        max_prob_bin = torch.topk(soft_assignment,1).indices

        # print("Max Prob", max_prob_bin[0,0,200,:])
        hard_assignment = torch.reshape(F.one_hot(max_prob_bin, self.num_kmean_kernels),
                                     (input_size[0], self.feat_maps, self.code_length, self.num_kmean_kernels))
        # print('hard_assignment', hard_assignment[0,0,200,:])
        # print('soft_assignment', soft_assignment.size())

        # If training, soft assignment, else hard
        assignment = soft_assignment if self.training else hard_assignment

        bit_code = torch.matmul(assignment,self.bins)
        
        return bit_code     