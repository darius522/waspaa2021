from torch.nn import LSTM, Linear, BatchNorm1d, Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F

class Waveunet(nn.Module):
    def __init__(
        self,
        n_frames = 16384,
        num_layers = 12, # How many U-Net layers
        filter_size = 15, # Filter size of conv in downsampling block
        num_filters = 24,
        kernel_size_down = 15,
        kernel_size_up = 5,
        output_filter_size = 1, # Filter size of convolution in the output layer
        output_activation  = 'tanh',
        strides = 1
    ):
        """
        Input: (nb_samples, nb_in_channels)
        Output: (nb_samples, nb_out_channels)
        """

        super(Waveunet, self).__init__()

        x = inputs

        # Encoding Path
        for layer in range(num_layers):

            num_filters_layer = num_filters + (num_filters * layer)
            out_frames = out_frames / 2

            x = nn.Conv1D(
                in_channels=self.n_frames,
                out_channels=out_frames,
                filters=num_filters_layer, 
                kernel_size=self.kernel_size_down,
                padding=(self.kernel_size_down // 2),
                strides=self.strides)(x)

            x = LeakyReLU(alpha=0.2)(x)
            x = BatchNorm1d()(x)

            # Save skip connection for decoding path and downsample
            skip_connections.append(x)
            x = Lambda(downsample)(x)

        x = Conv1D(
            filters=num_filters + (num_filters * num_layers), 
            kernel_size=kernel_size_down,
            padding='same',
            strides=strides)(x)

        # Decoding Path
        for layer in range(num_layers):

            skip_layer = skip_connections[-layer-1]
            
            # Makesure that num_filter coincide with current layer shape
            num_filter = skip_layer.get_shape().as_list()[2]

            # Upsample and Concatenate
            x = Lambda(upsample)(x)
            x = Concatenate(axis=2)([x, skip_layer])

            x = Conv1D(
                filters=num_filter, 
                kernel_size=kernel_size_up,
                padding='same',
                strides=strides)(x)

            x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization()(x)

        # Last concat is with original input
        x = Concatenate(axis=2)([inputs, x])

        # Collapse feature maps into N sources
        out_bass   = Conv1D(filters=1, kernel_size=1, padding='same',activation='tanh')(x)
        out_drums  = Conv1D(filters=1, kernel_size=1, padding='same',activation='tanh')(x)
        out_other  = Conv1D(filters=1, kernel_size=1, padding='same',activation='tanh')(x)
        out_vocals = Conv1D(filters=1, kernel_size=1, padding='same',activation='tanh')(x)

        outputs = Concatenate(axis=-1)([out_bass,out_drums])
        outputs = Concatenate(axis=-1)([outputs,out_other])
        outputs = Concatenate(axis=-1)([outputs,out_vocals])