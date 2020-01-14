import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, act=None):
        super().__init__()
        self.net = nn.Sequential()

        self.net.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=((kernel_size - 1) // 2), groups=groups))
        if act == 'relu':
            self.net.add_module('relu', nn.ReLU(inplace=True))
    
    def forward(self, inputs):
        return self.net(inputs)

class conv_bn_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, act=None, bn=True, bias_attr=False):
        super().__init__()
        self.net = nn.Sequential()

        self.net.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=((kernel_size - 1) // 2), groups=groups, bias=bias_attr))
        if bn == True:
            self.net.add_module('bn', nn.BatchNorm2d(out_channels))
            if act and act == 'relu':
                self.net.add_module('relu', nn.ReLU(inplace=True))
    
    def forward(self, inputs):
        return self.net(inputs)

class bottleneck_block_layer(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.net = nn.Sequential()

        self.net.add_module('bbl_conv1', conv_bn_layer(in_channels, out_channels, kernel_size=1, act='relu'))
        self.net.add_module('bbl_conv2', conv_bn_layer(out_channels, out_channels, kernel_size=3, stride=stride, act=None))
    def forward(self, inputs):
        return self.net(inputs)

class encoder_block_layer(nn.Module):
    def __init__(self, in_channels, encoder_depth, encoder_filter):
        super().__init__()
        self.net = nn.Sequential()




class up_layer(nn.Module):
    def __init__(self, in_channels, concat_input, decoder_depths, decoder_filters, block):
        super().__init__()
        self.net = nn.UpsamplingBilinear2d(scale_factor=2)
    
    def forward(self, inputs):
        return self.net(inputs)

        
        
        


class encode_block_layer(nn.Module):
    def __init__(self, in_channels, encoder_depths, encoder_filters, block):
        super().__init__()
        self.net = nn.Sequential()

        for index in range(encoder_depths[block]):
            self.net.add_module('encoder_block{}_conv1'.format(block), conv_bn_layer(in_channels, encoder_filters[block], kernel_size=1, act='relu'))
            self.net.add_module('encoder_block{}_conv1'.format(block), conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=3, stride=stride, act='relu'))

    def forward(self, inputs):
        self.net(inputs)