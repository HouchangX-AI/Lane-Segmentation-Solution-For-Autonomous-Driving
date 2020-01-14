import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from models.unet_party import conv_bn_layer,conv_layer


class unet_base(nn.Module):
    def __init__(self, label_number, img_size):
        super().__init__()
        encoder_depths = [3, 4, 6, 4]
        encoder_filters = [64, 128, 256, 512]
        decoder_depth = [4, 3, 3, 2]
        decoder_filters = [256, 128, 64, 32]
        
        self.pre_conv1 = conv_bn_layer(3, 32, 3, 2, act='relu')
        self.pre_conv2 = conv_bn_layer(32, 32, 3, act='relu')
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ## encoder step1 循环 block=0
        block = 0
        ## for i in range(encoder_depths[0]):  i=0 stride=1 out_channel=64
        self.encoder_block0_conv1 = conv_bn_layer(32, encoder_filters[block], kernel_size=1, act='relu')
        self.encoder_block0_conv2 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=3)
        self.encoder_block0_convert1 = conv_bn_layer(32, encoder_filters[block], kernel_size=1)
        ## TODO 逐元素相加 shape=64
        ## for i in range(encoder_depths[0]):  i=1
        self.encoder_block0_conv3 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=1, act='relu')
        self.encoder_block0_conv4 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=3)
        ## TODO 逐元素相加 channel=64
        ## for i in range(encoder_depths[0])   i=2
        self.encoder_block0_conv5 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=1, act='relu')
        self.encoder_block0_conv6 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=3)
        ## TODO 逐元素相加 channel=64  conv0

        ## encoder step2循环 block=1
        block = 1
        ## for i in range(encoder_depths[1])   i=0
        self.encoder_block1_conv1 = conv_bn_layer(64, encoder_filters[block], kernel_size=1, act='relu')
        self.encoder_block1_conv2 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=3, stride=2)
        self.encoder_block1_convert1 = conv_bn_layer(64, encoder_filters[block], kernel_size=1, stride=2)
        ## TODO 逐元素相加 channel=128
        ## for i in range(encoder_depths[1])   i=1
        self.encoder_block1_conv3 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=1, act='relu')
        self.encoder_block1_conv4 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=3)
        ## TODO 逐元素相加 channel=128
        ## for i in range(encoder_depths[1])   i=2
        self.encoder_block1_conv5 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=1, act='relu')
        self.encoder_block1_conv6 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=3)
        ## TODO 逐元素相加 channel=128
        ## for i in range(encoder_depths[1])   i=3
        self.encoder_block1_conv7 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=1, act='relu')
        self.encoder_block1_conv8 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=3)
        ## TODO 逐元素相加 channel=128   conv1

        ## encoder step3循环 block=2
        block=2
        ## for i in range(encoder_depths[2])   i=0
        self.encoder_block2_conv1 = conv_bn_layer(128, encoder_filters[block], kernel_size=1, act='relu')
        self.encoder_block2_conv2 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=3, stride=2)
        self.encoder_block2_convert1 = conv_bn_layer(128, encoder_filters[block], kernel_size=1, stride=2)
        ## TODO 逐元素相加 channel=256
        ## for i in range(encoder_depths[2])   i=1
        self.encoder_block2_conv3 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=1, act='relu')
        self.encoder_block2_conv4 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=3)
        ## TODO 逐元素相加 channel=256
        ## for i in range(encoder_depths[2])   i=2
        self.encoder_block2_conv5 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=1, act='relu')
        self.encoder_block2_conv6 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=3)
        ## TODO 逐元素相加 channel=256
        ## for i in range(encoder_depths[2])   i=3
        self.encoder_block2_conv7 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=1, act='relu')
        self.encoder_block2_conv8 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=3)
        ## TODO 逐元素相加 channel=256
        ## for i in range(encoder_depths[2])   i=4
        self.encoder_block2_conv9 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=1, act='relu')
        self.encoder_block2_conv10 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=3)
        ## TODO 逐元素相加 channel=256
        ## for i in range(encoder_depths[2])   i=5
        self.encoder_block2_conv11 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=1, act='relu')
        self.encoder_block2_conv12 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=3)
        ## TODO 逐元素相加 channel=256   conv2

        ## encoder step4循环 block=3
        block=3
        ## for i in range(encoder_depths[3])   i=0
        self.encoder_block3_conv1 = conv_bn_layer(256, encoder_filters[block], kernel_size=1, act='relu')
        self.encoder_block3_conv2 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=3, stride=2)
        self.encoder_block3_convert1 = conv_bn_layer(256, encoder_filters[block], kernel_size=1, stride=2)
        ## TODO 逐元素相加 channel=512
        ## for i in range(encoder_depths[3])   i=1
        self.encoder_block3_conv3 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=1, act='relu')
        self.encoder_block3_conv4 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=3)
        ## TODO 逐元素相加 channel=512
        ## for i in range(encoder_depths[3])   i=2
        self.encoder_block3_conv5 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=1, act='relu')
        self.encoder_block3_conv6 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=3)
        ## TODO 逐元素相加 channel=512
        ## for i in range(encoder_depths[3])   i=3
        self.encoder_block3_conv7 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=1, act='relu')
        self.encoder_block3_conv8 = conv_bn_layer(encoder_filters[block], encoder_filters[block], kernel_size=3)
        ## TODO 逐元素相加 channel=512  conv3


        ## decoder step1 block=0
        block=0
        self.decoder_block0_upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.decoder_block0_conv1 = conv_bn_layer(512, decoder_filters[block], kernel_size=1, act='relu')
        self.decoder_block0_conv2 = conv_bn_layer(decoder_filters[block], decoder_filters[block], kernel_size=3)
        self.decoder_block0_convert = conv_bn_layer(512, decoder_filters[block], kernel_size=1)
        ## TODO 逐元素相加 channel=256  decode_bn1
        ## conv2
        self.decoder_block0_concat = conv_bn_layer(256, 128, kernel_size=1, act='relu')
        ## TODO 拼接把decoder_block0_convert的结果和decoder_block0_concat的结果相加 shape=128+256=384
        ## decoder step1循环
        ## for i in range(decode_depths[0])  i=0
        self.decoder_block0_conv3 = conv_bn_layer(384, decoder_filters[block], kernel_size=1, act='relu')
        self.decoder_block0_conv4 = conv_bn_layer(decoder_filters[block], decoder_filters[block], kernel_size=3)
        self.decoder_block0_convert1 = conv_bn_layer(384, decoder_filters[block], kernel_size=1)
        ## TODO 逐元素相加 channel=256
        ## for i in range(decode_depths[0])  i=1
        self.decoder_block0_conv5 = conv_bn_layer(decoder_filters[block], decoder_filters[block], kernel_size=1, act='relu')
        self.decoder_block0_conv6 = conv_bn_layer(decoder_filters[block], decoder_filters[block], kernel_size=3)
        ## TODO 逐元素相加 channel=256
        ## for i in range(decode_depths[0])  i=2
        self.decoder_block0_conv7 = conv_bn_layer(decoder_filters[block], decoder_filters[block], kernel_size=1, act='relu')
        self.decoder_block0_conv8 = conv_bn_layer(decoder_filters[block], decoder_filters[block], kernel_size=3)
        ## TODO 逐元素相加 channel=256
        ## for i in range(decode_depths[0])  i=3
        self.decoder_block0_conv9 = conv_bn_layer(decoder_filters[block], decoder_filters[block], kernel_size=1, act='relu')
        self.decoder_block0_conv10 = conv_bn_layer(decoder_filters[block], decoder_filters[block], kernel_size=3)
        ## TODO 逐元素相加 channel=256

        ## decoder step2 block=1 out_channel=128
        block=1
        self.decoder_block1_upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.decoder_block1_conv1 = conv_bn_layer(256, decoder_filters[block], kernel_size=1, act='relu')
        self.decoder_block1_conv2 = conv_bn_layer(decoder_filters[block], decoder_filters[block], kernel_size=3)
        self.decoder_block1_convert = conv_bn_layer(256, decoder_filters[block], kernel_size=1)
        ## TODO 逐元素相加 channel=128
        self.decoder_block1_concat = conv_bn_layer(128, 64, kernel_size=1, act='relu')
        ## TODO 拼接把decoder_block1_convert的结果和decoder_block1_concat的结果相加 shape=128+64=192

        ## decoder step2循环
        ## for i in range(decode_depths[0])  i=0
        self.decoder_block1_conv3 = conv_bn_layer(192, decoder_filters[block], kernel_size=1, act='relu')
        self.decoder_block1_conv4 = conv_bn_layer(decoder_filters[block], decoder_filters[block], kernel_size=3)
        self.decoder_block1_convert1 = conv_bn_layer(192, decoder_filters[block], kernel_size=1)
        ## TODO 逐元素相加 channel=128
        ## for i in range(decode_depths[0])  i=1
        self.decoder_block1_conv5 = conv_bn_layer(decoder_filters[block], decoder_filters[block], kernel_size=1, act='relu')
        self.decoder_block1_conv6 = conv_bn_layer(decoder_filters[block], decoder_filters[block], kernel_size=3)
        ## TODO 逐元素相加 channel=128
        ## for i in range(decode_depths[0])  i=2
        self.decoder_block1_conv7 = conv_bn_layer(decoder_filters[block], decoder_filters[block], kernel_size=1, act='relu')
        self.decoder_block1_conv8 = conv_bn_layer(decoder_filters[block], decoder_filters[block], kernel_size=3)
        ## TODO 逐元素相加 channel=128

        ## decoder step3 block=2 out_channel=64
        block=2
        self.decoder_block2_upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.decoder_block2_conv1 = conv_bn_layer(128, decoder_filters[block], kernel_size=1, act='relu')
        self.decoder_block2_conv2 = conv_bn_layer(decoder_filters[block], decoder_filters[block], kernel_size=3)
        self.decoder_block2_convert = conv_bn_layer(128, decoder_filters[block], kernel_size=1)
        ## TODO 逐元素相加 channel=64
        self.decoder_block2_concat = conv_bn_layer(64, 32, kernel_size=1, act='relu') 
        ## TODO 拼接把decoder_block2_convert的结果和decoder_block2_concat的结果相加 shape=32+64=96

        ## decoder step3循环
        ## for i in range(decode_depths[0])  i=0
        self.decoder_block2_conv3 = conv_bn_layer(96, decoder_filters[block], kernel_size=1, act='relu')
        self.decoder_block2_conv4 = conv_bn_layer(decoder_filters[block], decoder_filters[block], kernel_size=3)
        self.decoder_block2_convert1 = conv_bn_layer(96, decoder_filters[block], kernel_size=1)
        ## TODO 逐元素相加 channel=64
        ## for i in range(decode_depths[0])  i=1
        self.decoder_block2_conv5 = conv_bn_layer(decoder_filters[block], decoder_filters[block], kernel_size=1, act='relu')
        self.decoder_block2_conv6 = conv_bn_layer(decoder_filters[block], decoder_filters[block], kernel_size=3)
        ## TODO 逐元素相加 channel=64
        ## for i in range(decode_depths[0])  i=2
        self.decoder_block2_conv7 = conv_bn_layer(decoder_filters[block], decoder_filters[block], kernel_size=1, act='relu')
        self.decoder_block2_conv8 = conv_bn_layer(decoder_filters[block], decoder_filters[block], kernel_size=3)
        ## TODO 逐元素相加 channel=64

        ## decoder step4 block=3 out_channel=32
        block=3
        self.decoder_block3_upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.decoder_block3_conv1 = conv_bn_layer(64, decoder_filters[block], kernel_size=1, act='relu')
        self.decoder_block3_conv2 = conv_bn_layer(decoder_filters[block], decoder_filters[block], kernel_size=3)
        self.decoder_block3_convert = conv_bn_layer(64, decoder_filters[block], kernel_size=1)
        ## TODO 逐元素相加 channel=32
        self.decoder_block3_concat = conv_bn_layer(32, 16, kernel_size=1, act='relu') 
        ## TODO 拼接把decoder_block3_convert的结果和decoder_block3_concat的结果相加 shape=32+16=48

        ## decoder step3循环
        ## for i in range(decode_depths[0])  i=0
        self.decoder_block3_conv3 = conv_bn_layer(48, decoder_filters[block], kernel_size=1, act='relu')
        self.decoder_block3_conv4 = conv_bn_layer(decoder_filters[block], decoder_filters[block], kernel_size=3)
        self.decoder_block3_convert1 = conv_bn_layer(48, decoder_filters[block], kernel_size=1)
        ## TODO 逐元素相加 channel=32
        ## for i in range(decode_depths[0])  i=1
        self.decoder_block3_conv5 = conv_bn_layer(decoder_filters[block], decoder_filters[block], kernel_size=1, act='relu')
        self.decoder_block3_conv6 = conv_bn_layer(decoder_filters[block], decoder_filters[block], kernel_size=3)
        ## TODO 逐元素相加 channel=32

        ## TODO 双线性改变输出变成原图大小
        self.decoder_block4_upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.decoder_block4_conv1 = conv_bn_layer(32, 32, kernel_size=1, act='relu')
        self.decoder_block4_conv2 = conv_bn_layer(32, 32, kernel_size=3)
        ## TODO 逐元素相加 channel=32
        self.decoder_block4_conv3 = conv_bn_layer(32, 16, kernel_size=1, act='relu')
        self.decoder_block4_conv4 = conv_bn_layer(16, 16, kernel_size=3)
        self.decoder_block4_convert = conv_bn_layer(32, 16, kernel_size=1)
        ## TODO 逐元素相加 channel=16

        ## label_number
        self.logit = conv_layer(16, label_number, kernel_size=1, act=None)

    
    def forward(self, inputs):
        pre_conv1 = self.pre_conv1(inputs)
        pre_conv2 = self.pre_conv2(pre_conv1)
        maxpool1 = self.maxpool1(pre_conv2)

        encoder_block0_conv1 = self.encoder_block0_conv1(maxpool1)
        encoder_block0_conv2 = self.encoder_block0_conv2(encoder_block0_conv1)
        encoder_block0_convert1 = self.encoder_block0_convert1(maxpool1)
        ## 逐元素相加
        encoder_block0_convert1 = encoder_block0_convert1 + encoder_block0_conv2

        encoder_block0_conv3 = self.encoder_block0_conv3(encoder_block0_convert1)
        encoder_block0_conv4 = self.encoder_block0_conv4(encoder_block0_conv3)
        ## 逐元素相加
        encoder_block0_conv4 = encoder_block0_conv4 + encoder_block0_convert1

        encoder_block0_conv5 = self.encoder_block0_conv5(encoder_block0_conv4)
        encoder_block0_conv6 = self.encoder_block0_conv6(encoder_block0_conv5)
        ## 逐元素相加
        encoder_block0_conv6 = encoder_block0_conv6 + encoder_block0_conv4

        encoder_block1_conv1 = self.encoder_block1_conv1(encoder_block0_conv6)
        encoder_block1_conv2 = self.encoder_block1_conv2(encoder_block1_conv1)
        encoder_block1_convert1 = self.encoder_block1_convert1(encoder_block0_conv6)
        ## 逐元素相加
        encoder_block1_convert1 = encoder_block1_convert1 + encoder_block1_conv2

        encoder_block1_conv3 = self.encoder_block1_conv3(encoder_block1_convert1)
        encoder_block1_conv4 = self.encoder_block1_conv4(encoder_block1_conv3)
        ## 逐元素相加
        encoder_block1_conv4 = encoder_block1_conv4 + encoder_block1_convert1

        encoder_block1_conv5 = self.encoder_block1_conv5(encoder_block1_conv4)
        encoder_block1_conv6 = self.encoder_block1_conv6(encoder_block1_conv5)
        ## 逐元素相加
        encoder_block1_conv6 = encoder_block1_conv6 + encoder_block1_conv4

        encoder_block1_conv7 = self.encoder_block1_conv7(encoder_block1_conv6)
        encoder_block1_conv8 = self.encoder_block1_conv8(encoder_block1_conv7)
        ## 逐元素相加
        encoder_block1_conv8 = encoder_block1_conv8 + encoder_block1_conv6

        encoder_block2_conv1 = self.encoder_block2_conv1(encoder_block1_conv8)
        encoder_block2_conv2 = self.encoder_block2_conv2(encoder_block2_conv1)
        encoder_block2_convert1 = self.encoder_block2_convert1(encoder_block1_conv8)
        ## 主元素相加
        encoder_block2_convert1 = encoder_block2_convert1 + encoder_block2_conv2

        encoder_block2_conv3 = self.encoder_block2_conv3(encoder_block2_convert1)
        encoder_block2_conv4 = self.encoder_block2_conv4(encoder_block2_conv3)
        ## 逐元素相加
        encoder_block2_conv4 = encoder_block2_conv4 + encoder_block2_convert1

        encoder_block2_conv5 = self.encoder_block2_conv5(encoder_block2_conv4)
        encoder_block2_conv6 = self.encoder_block2_conv6(encoder_block2_conv5)
        ## 逐元素相加
        encoder_block2_conv6 = encoder_block2_conv6 + encoder_block2_conv4

        encoder_block2_conv7 = self.encoder_block2_conv7(encoder_block2_conv6)
        encoder_block2_conv8 = self.encoder_block2_conv8(encoder_block2_conv7)
        ## 逐元素相加
        encoder_block2_conv8 = encoder_block2_conv8 + encoder_block2_conv6

        encoder_block2_conv9 = self.encoder_block2_conv9(encoder_block2_conv8)
        encoder_block2_conv10 = self.encoder_block2_conv10(encoder_block2_conv9)
        ## 主元素相加
        encoder_block2_conv10 = encoder_block2_conv10 + encoder_block2_conv8

        encoder_block2_conv11 = self.encoder_block2_conv11(encoder_block2_conv10)
        encoder_block2_conv12 = self.encoder_block2_conv12(encoder_block2_conv11)
        ## 逐元素相加
        encoder_block2_conv12 = encoder_block2_conv12 + encoder_block2_conv10

        encoder_block3_conv1 = self.encoder_block3_conv1(encoder_block2_conv12)
        encoder_block3_conv2 = self.encoder_block3_conv2(encoder_block3_conv1)
        encoder_block3_convert1 = self.encoder_block3_convert1(encoder_block2_conv12)
        ## 逐元素相加
        encoder_block3_convert1 = encoder_block3_convert1 + encoder_block3_conv2

        encoder_block3_conv3 = self.encoder_block3_conv3(encoder_block3_convert1)
        encoder_block3_conv4 = self.encoder_block3_conv4(encoder_block3_conv3)
        ## 逐元素相加
        encoder_block3_conv4 = encoder_block3_conv4 + encoder_block3_convert1

        encoder_block3_conv5 = self.encoder_block3_conv5(encoder_block3_conv4)
        encoder_block3_conv6 = self.encoder_block3_conv6(encoder_block3_conv5)
        ## 逐元素相加
        encoder_block3_conv6 = encoder_block3_conv6 + encoder_block3_conv4

        encoder_block3_conv7 = self.encoder_block3_conv7(encoder_block3_conv6)
        encoder_block3_conv8 = self.encoder_block3_conv8(encoder_block3_conv7)
        ## 逐元素相加
        encoder_block3_conv8 = encoder_block3_conv8 + encoder_block3_conv6

        decoder_block0_upsampling = self.decoder_block0_upsampling(encoder_block3_conv8)
        decoder_block0_conv1 = self.decoder_block0_conv1(decoder_block0_upsampling)
        decoder_block0_conv2 = self.decoder_block0_conv2(decoder_block0_conv1)
        decoder_block0_convert = self.decoder_block0_convert(decoder_block0_upsampling)
        ## 逐元素相加
        decoder_block0_convert = decoder_block0_convert = decoder_block0_conv2

        decoder_block0_concat = self.decoder_block0_concat(encoder_block2_conv12)
        ## 拼接把decoder_block0_convert的结果和decoder_block0_concat的结果相加 shape=128+256=384
        decoder_block0_cat = torch.cat((decoder_block0_concat, decoder_block0_convert), 1)

        decoder_block0_conv3 = self.decoder_block0_conv3(decoder_block0_cat)
        decoder_block0_conv4 = self.decoder_block0_conv4(decoder_block0_conv3)
        decoder_block0_convert1 = self.decoder_block0_convert1(decoder_block0_cat)
        ## 逐元素相加
        decoder_block0_convert1 = decoder_block0_convert1 + decoder_block0_conv4

        decoder_block0_conv5 = self.decoder_block0_conv5(decoder_block0_convert1)
        decoder_block0_conv6 = self.decoder_block0_conv6(decoder_block0_conv5)
        ## 逐元素相加
        decoder_block0_conv6 = decoder_block0_conv6 + decoder_block0_convert1

        decoder_block0_conv7 = self.decoder_block0_conv7(decoder_block0_conv6)
        decoder_block0_conv8 = self.decoder_block0_conv8(decoder_block0_conv7)
        ## 逐元素相加
        decoder_block0_conv8 = decoder_block0_conv8 + decoder_block0_conv6

        decoder_block0_conv9 = self.decoder_block0_conv9(decoder_block0_conv8)
        decoder_block0_conv10 = self.decoder_block0_conv10(decoder_block0_conv9)
        ## 逐元素相加
        decoder_block0_conv10 = decoder_block0_conv10 + decoder_block0_conv8

        decoder_block1_upsampling = self.decoder_block1_upsampling(decoder_block0_conv10)
        decoder_block1_conv1 = self.decoder_block1_conv1(decoder_block1_upsampling)
        decoder_block1_conv2 = self.decoder_block1_conv2(decoder_block1_conv1)
        decoder_block1_convert = self.decoder_block1_convert(decoder_block1_upsampling)
        ## 逐元素相加
        decoder_block1_convert = decoder_block1_convert + decoder_block1_conv2
        ## 拼接把decoder_block1_convert的结果和decoder_block1_concat的结果相加 shape=128+64=192
        decoder_block1_concat = self.decoder_block1_concat(encoder_block1_conv8)
        decoder_block1_cat = torch.cat((decoder_block1_concat, decoder_block1_convert), 1)

        decoder_block1_conv3 = self.decoder_block1_conv3(decoder_block1_cat)
        decoder_block1_conv4 = self.decoder_block1_conv4(decoder_block1_conv3)
        decoder_block1_convert1 = self.decoder_block1_convert1(decoder_block1_cat)
        ## 逐元素相加
        decoder_block1_convert1 = decoder_block1_convert1 + decoder_block1_conv4
        
        decoder_block1_conv5 = self.decoder_block1_conv5(decoder_block1_convert1)
        decoder_block1_conv6 = self.decoder_block1_conv6(decoder_block1_conv5)
        ## 逐元素相加
        decoder_block1_conv6 = decoder_block1_conv6 + decoder_block1_convert1

        decoder_block1_conv7 = self.decoder_block1_conv7(decoder_block1_conv6)
        decoder_block1_conv8 = self.decoder_block1_conv8(decoder_block1_conv7)
        ## 逐元素相加
        decoder_block1_conv8 = decoder_block1_conv8 + decoder_block1_conv6

        decoder_block2_upsampling = self.decoder_block2_upsampling(decoder_block1_conv8)

        decoder_block2_conv1 = self.decoder_block2_conv1(decoder_block2_upsampling)
        decoder_block2_conv2 = self.decoder_block2_conv2(decoder_block2_conv1)
        decoder_block2_convert = self.decoder_block2_convert(decoder_block2_upsampling)
        ## 逐元素相加
        decoder_block2_convert = decoder_block2_convert + decoder_block2_conv2
        decoder_block2_concat = self.decoder_block2_concat(encoder_block0_conv6)

        decoder_block2_cat = torch.cat((decoder_block2_concat, decoder_block2_convert), 1)

        decoder_block2_conv3 = self.decoder_block2_conv3(decoder_block2_cat)
        decoder_block2_conv4 = self.decoder_block2_conv4(decoder_block2_conv3)
        decoder_block2_convert1 = self.decoder_block2_convert1(decoder_block2_cat)
        ## 逐元素相加
        decoder_block2_convert1 = decoder_block2_convert1 + decoder_block2_conv4

        decoder_block2_conv5 = self.decoder_block2_conv5(decoder_block2_convert1)
        decoder_block2_conv6 = self.decoder_block2_conv6(decoder_block2_conv5)
        ## 逐元素相加
        decoder_block2_conv6 = decoder_block2_conv6 + decoder_block2_convert1

        decoder_block2_conv7 = self.decoder_block2_conv7(decoder_block2_conv6)
        decoder_block2_conv8 = self.decoder_block2_conv8(decoder_block2_conv7)
        ## 逐元素相加
        decoder_block2_conv8 = decoder_block2_conv8 + decoder_block2_conv6

        decoder_block3_upsampling = self.decoder_block3_upsampling(decoder_block2_conv8)
        decoder_block3_conv1 = self.decoder_block3_conv1(decoder_block3_upsampling)
        decoder_block3_conv2 = self.decoder_block3_conv2(decoder_block3_conv1)
        decoder_block3_convert = self.decoder_block3_convert(decoder_block3_upsampling)
        ## 逐元素相加
        decoder_block3_convert = decoder_block3_convert + decoder_block3_conv2
        decoder_block3_concat = self.decoder_block3_concat(pre_conv2)
        decoder_block3_cat = torch.cat((decoder_block3_concat, decoder_block3_convert), 1)

        decoder_block3_conv3 = self.decoder_block3_conv3(decoder_block3_cat)
        decoder_block3_conv4 = self.decoder_block3_conv4(decoder_block3_conv3)
        decoder_block3_convert1 = self.decoder_block3_convert1(decoder_block3_cat)
        ## 逐元素相加
        decoder_block3_convert1 = decoder_block3_convert1 + decoder_block3_conv4

        decoder_block3_conv5 = self.decoder_block3_conv5(decoder_block3_convert1)
        decoder_block3_conv6 = self.decoder_block3_conv6(decoder_block3_conv5)
        ## 逐元素相加
        decoder_block3_conv6 = decoder_block3_conv6 + decoder_block3_convert1

        # print ('decoder之后尺寸:', decoder_block3_conv6.shape)


        decoder_block4_upsampling = self.decoder_block4_upsampling(decoder_block3_conv6)
        decoder_block4_conv1 = self.decoder_block4_conv1(decoder_block4_upsampling)
        decoder_block4_conv2 = self.decoder_block4_conv2(decoder_block4_conv1)
        ## 逐元素相加
        decoder_block4_conv2 = decoder_block4_conv2 + decoder_block4_upsampling

        decoder_block4_conv3 = self.decoder_block4_conv3(decoder_block4_conv2)
        decoder_block4_conv4 = self.decoder_block4_conv4(decoder_block4_conv3)
        decoder_block4_convert = self.decoder_block4_convert(decoder_block4_conv2)
        ## 逐元素相加
        decoder_block4_convert = decoder_block4_convert + decoder_block4_conv4

        logit = self.logit(decoder_block4_convert)

        # print("| Output Predictions:", logit.size())
        # print("| Final Predictions:", logit.size())

        return logit

