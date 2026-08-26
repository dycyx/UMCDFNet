import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm

from .CAFAM import CAFA
from .SDFTM import SDFT
from .FDFTM import FDFT
from .MDIFM import MDIF
from .SF2GLM import SF2GL


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class FFM(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(FFM, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        else:
            m.initialize()

class Enhance(nn.Module):
    def __init__(self,in_c):
        super(Enhance, self).__init__()
        self.conv0 = nn.Conv2d(in_c,in_c, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(in_c)

    def forward(self, input1, input2):
        out = F.relu(self.bn0(self.conv0(input1 + input2)), inplace=True)
        return out

    def initialize(self):
        weight_init(self)

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.PReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Split(nn.Module):
    def __init__(self,c_in=256, c_out=128):
        super(Split, self).__init__()
        self.br2 = nn.Sequential(
            BasicConv(c_in, c_out, kernel_size=1, bias=False, bn=True, relu=True),

            BasicConv(c_out, c_out, kernel_size=3, dilation=1, padding=1, groups=c_out, bias=False,
                      relu=False),
        )

        self.conv1b = nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1)
        self.bn1b = nn.BatchNorm2d(c_out)

        self.conv1d = nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1)
        self.bn1d = nn.BatchNorm2d(c_out)


    def forward(self, x):

        out2 = self.br2(x)
        out1b = F.relu(self.bn1b(self.conv1b(out2)), inplace=True)
        out1d = F.relu(self.bn1d(self.conv1d(out2)), inplace=True)

        return out1b, out1d


class UMCDFNet(nn.Module):
    def __init__(self,
                 decode_channels=96,
                 dropout=0.1,
                 backbone_name="convnext_base_384_in22ft1k",
                 # backbone_name="resnet101d",
                 # backbone_name="swin_base_patch4_window12_384_in22k",
                 # backbone_name="vgg16",
                 # backbone_name="convnext_tiny.in12k_ft_in1k_384,convnextv2_base.fcmae_ft_in22k_in1k_384",
                 pretrained=True,
                 window_size=8,
                 ):
        super().__init__()
        self.backbone = timm.create_model(model_name=backbone_name, features_only=True, pretrained=pretrained,
                                          output_stride=32, out_indices=(0, 1, 2, 3))

        self.cafa = CAFA(1024, num_heads=2)

        self.spafuse = SDFT(in_dim=3 * decode_channels, out_dim=decode_channels)
        
        self.frefuse = FDFT(3 * decode_channels)

        self.feature_bd = MDIF(decode_channels, num_heads=8, LayerNorm_type='WithBias')
        self.feature_dt = MDIF(decode_channels, num_heads=8, LayerNorm_type='WithBias')
        
        self.SF2GL = SF2GL(in_ch=decode_channels*2, out_ch=decode_channels)
        
        self.ffm1 = FFM(in_channels=decode_channels, decode_channels=decode_channels)
        self.ffm2 = FFM(in_channels=decode_channels, decode_channels=decode_channels)

        self.split = Split(c_in=decode_channels*2, c_out=decode_channels)
        self.enhance_b = Enhance(in_c=decode_channels)
        self.enhance_d = Enhance(in_c=decode_channels)

        self.down1 = BasicConv(3 * decode_channels, decode_channels, kernel_size=3, dilation=1, padding=1, groups=decode_channels, bias=False,
                      relu=False)
        self.down2 = BasicConv(3 * decode_channels, decode_channels, kernel_size=3, dilation=1, padding=1, groups=decode_channels, bias=False,
                      relu=False)

        self.linearb1 = nn.Conv2d(decode_channels, 1, kernel_size=3, padding=1)
        self.lineard1 = nn.Conv2d(decode_channels, 1, kernel_size=3, padding=1)
        self.linearb2 = nn.Conv2d(decode_channels, 1, kernel_size=3, padding=1)
        self.lineard2= nn.Conv2d(decode_channels, 1, kernel_size=3, padding=1)
        self.linear1 = nn.Sequential(nn.Conv2d(decode_channels*2, decode_channels, kernel_size=3, padding=1), nn.BatchNorm2d(decode_channels),
                                    nn.ReLU(inplace=True), nn.Conv2d(decode_channels, 1, kernel_size=3, padding=1))
        self.linear2 = nn.Sequential(nn.Conv2d(decode_channels * 2, decode_channels, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(decode_channels),
                                    nn.ReLU(inplace=True), nn.Conv2d(decode_channels, 1, kernel_size=3, padding=1))

    def forward(self, x, y, imagename=None):

        res1, res2, res3, res4 = self.backbone(x)
        tes1, tes2, tes3, tes4 = self.backbone(y)
        # torch.Size([8, 128, 96, 96])
        # torch.Size([8, 256, 48, 48])
        # torch.Size([8, 512, 24, 24])
        # torch.Size([8, 1024, 12, 12])

        # 对齐模块
        res1, tes1, mid_res, mid_tes = self.cafa(res1, res2, res3, res4, tes1, tes2, tes3, tes4)
        # res1 torch.Size([8, 96, 96, 96])
        # tes1 torch.Size([8, 96, 96, 96])
        # mid_res torch.Size([8, 288, 96, 96])
        # mid_tes torch.Size([8, 288, 96, 96])

        spa_fuse = self.spafuse(mid_res,mid_tes) # torch.Size([8, 192, 96, 96])
        
        fre_l , fre_h = self.frefuse(mid_res , mid_tes) # torch.Size([8, 96, 96, 96])

        glb, local = self.SF2GL(spa_fuse, imagename)# torch.Size([8, 96, 96, 96])

        body = self.feature_bd(fre_l, glb) # torch.Size([8, 96, 96, 96])
        detail = self.feature_dt(fre_h,local) # torch.Size([8, 96, 96, 96])

        mid_res1 = mid_res
        mid_tes1 = mid_tes
        mid_res = self.down1(mid_res) # torch.Size([8, 96, 96, 96])
        mid_tes = self.down2(mid_tes) # torch.Size([8, 96, 96, 96])
        

        res = mid_res + body
        res = self.ffm1(res, res1) # torch.Size([8, 96, 96, 96])

        tes = mid_tes + detail
        tes = self.ffm2(tes, tes1) # torch.Size([8, 96, 96, 96])

        out1 = torch.cat([res, tes], dim=1)
        outb2, outd2 = self.split(out1)
        outb2 = self.enhance_b(res, outb2)
        outd2 = self.enhance_d(tes, outd2)
        out2 = torch.cat([outb2, outd2], dim=1)

        shape = x.size()[2:]
        out1 = F.interpolate(self.linear1(out1), size=shape, mode='bilinear')
        outb1 = F.interpolate(self.linearb1(res), size=shape, mode='bilinear')
        outd1 = F.interpolate(self.lineard1(tes), size=shape, mode='bilinear')

        out2 = F.interpolate(self.linear2(out2), size=shape, mode='bilinear')
        outb2 = F.interpolate(self.linearb2(outb2), size=shape, mode='bilinear')
        outd2 = F.interpolate(self.lineard2(outd2), size=shape, mode='bilinear')

        return outb1, outd1, out1, outb2, outd2, out2

if __name__ == '__main__':
    a = np.random.random((1, 3, 384, 384))
    b = np.random.random((1, 3, 384, 384))
    c = torch.Tensor(a).cuda()
    d = torch.Tensor(b).cuda()
    data = {'image': c, 'depth': d}
    net = UMCDFNet().cuda()
    out = net(c, d)

