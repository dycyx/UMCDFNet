# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class BasicConv(nn.Module):

#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
#         super(BasicConv, self).__init__()
#         self.out_channels = out_planes
#         self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
#         self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
#         self.relu = nn.PReLU() if relu else None

#     def forward(self, x):
#         x = self.conv(x)
#         if self.bn is not None:
#             x = self.bn(x)
#         if self.relu is not None:
#             x = self.relu(x)
#         return x

# class CBRBlock(nn.Sequential):
#     def __init__(self, in_c, out_c, kernel_size=3):
#         super().__init__()
#         padding = kernel_size // 2
#         self.add_module("conv", nn.Conv2d(in_c, out_c, kernel_size, padding=padding, bias=False))
#         self.add_module("bn", nn.BatchNorm2d(out_c))
#         self.add_module("act", nn.ReLU(inplace=True))

# class LargeKernelSpatialAttn(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=False)
#         self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3, bias=False)
#         self.conv1 = nn.Conv2d(dim, dim, 1, bias=False)
#         self.norm = nn.BatchNorm2d(dim)
        
#         self.cross_fuse = nn.Conv2d(dim * 2, dim, 1, bias=False)

#     def forward(self, x, y=None):
#         if y is not None:
#             combined = torch.cat([x, y], dim=1)
#             base_feat = self.cross_fuse(combined)
#         else:
#             base_feat = x
            
#         u = x
#         attn = self.conv0(base_feat)
#         attn = self.conv_spatial(attn)
#         attn = self.conv1(attn)
#         # Sigmoid 生成注意力图
#         attn_map = torch.sigmoid(attn)
        
#         return u * attn_map


# class CAFA(nn.Module):
#     def __init__(self, in_channels=1024, num_heads=4, decode_channels=96,decode_channels1=32):
#         super(CAFA, self).__init__()
#         self.in_channels = in_channels
#         self.num_heads = num_heads
  
#         self.query_conv = nn.Conv2d(in_channels, in_channels, 1, bias=False)
#         self.key_conv = nn.Conv2d(in_channels, in_channels, 1, bias=False)
#         self.value_conv = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        
#         self.spatial_cross_attn = LargeKernelSpatialAttn(in_channels)
        
#         self.weight_spatial = nn.Parameter(torch.zeros(1))
        
#         self.output_proj = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        
#         self.norm1 = nn.LayerNorm([in_channels, 1, 1])
#         self.norm_res = nn.BatchNorm2d(in_channels)
#         self.norm_ffn = nn.BatchNorm2d(in_channels)
        
#         self.ffn = nn.Sequential(
#             CBRBlock(in_channels, in_channels * 2, kernel_size=3),
#             nn.Conv2d(in_channels * 2, in_channels, 1, bias=False),
#             nn.BatchNorm2d(in_channels)
#         )
        
#         self.relu = nn.ReLU(inplace=True)
#         self.sigmoid = nn.Sigmoid()
        
#         self.conv1 = BasicConv(128, decode_channels, kernel_size=3, dilation=1, padding=1, groups=decode_channels1, bias=False,
#                       relu=False)
#         self.conv2 = BasicConv(256, decode_channels, kernel_size=3, dilation=1, padding=1, groups=decode_channels1, bias=False,
#                       relu=False)
#         self.conv3 = BasicConv(512, decode_channels, kernel_size=3, dilation=1, padding=1, groups=decode_channels1, bias=False,
#                       relu=False)
#         self.conv4 = BasicConv(1024, decode_channels, kernel_size=3, dilation=1, padding=1, groups=decode_channels1, bias=False,
#                       relu=False)

#         self.conv1_ = BasicConv(128, decode_channels, kernel_size=3, dilation=1, padding=1, groups=decode_channels1, bias=False,
#                       relu=False)
#         self.conv2_ = BasicConv(256, decode_channels, kernel_size=3, dilation=1, padding=1, groups=decode_channels1, bias=False,
#                       relu=False)
#         self.conv3_ = BasicConv(512, decode_channels, kernel_size=3, dilation=1, padding=1, groups=decode_channels1, bias=False,
#                       relu=False)
#         self.conv4_ = BasicConv(1024, decode_channels, kernel_size=3, dilation=1, padding=1, groups=decode_channels1, bias=False,
#                       relu=False)
      
#         self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
#         self.conv_sem3 = BasicConv(1024, 512, kernel_size=3, dilation=1, padding=1, groups=decode_channels1, bias=False, relu=False)
#         self.conv_sem2 = BasicConv(1024, 256, kernel_size=3, dilation=1, padding=1, groups=decode_channels1, bias=False, relu=False)


#     def forward(self, res1, res2, res3, res4, tes1, tes2, tes3, tes4):
#         B, C, H, W = res4.shape
#         res1h, res1w = res1.size()[-2:]  # 96,96
        
#         Q = self.query_conv(res4).view(B, self.num_heads, C // self.num_heads, H * W).permute(0, 1, 3, 2) # [B, nh, HW, hd]
#         K = self.key_conv(tes4).view(B, self.num_heads, C // self.num_heads, H * W).permute(0, 1, 3, 2)
#         V = self.value_conv(tes4).view(B, self.num_heads, C // self.num_heads, H * W).permute(0, 1, 3, 2)
        
#         attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / ((C // self.num_heads) ** 0.5)
#         attn_weights = F.softmax(attn_scores, dim=-1)
#         global_out = torch.matmul(attn_weights, V) # [B, nh, HW, hd]
#         global_out = global_out.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)
#         global_out = self.output_proj(global_out)
        
#         spatial_out = self.spatial_cross_attn(res4, tes4)
        
        
#         w_s = self.sigmoid(self.weight_spatial)
#         w_g = 1.0 - w_s
        
#         fused_attn = w_s * spatial_out + w_g * global_out
        
#         semantic = res4 + fused_attn
#         semantic = self.norm_res(semantic)
        
#         ffn_out = self.ffn(semantic)
#         out = semantic + ffn_out
#         out = self.norm_ffn(out)
#         outRGBT = self.relu(out)

#         res4 = res4*outRGBT
#         tes4 = tes4*outRGBT
#         res3 = res3*self.conv_sem3(self.up2(outRGBT))
#         tes3 = tes3*self.conv_sem3(self.up2(outRGBT))
#         res2 = res2*self.conv_sem2(self.up4(outRGBT))
#         tes2 = tes2 *self.conv_sem2(self.up4(outRGBT))

#         res1 = self.conv1(res1)
#         tes1 = self.conv1_(tes1)
#         res2 = self.conv2(res2)
#         tes2 = self.conv2_(tes2)
#         res3 = self.conv3(res3)
#         tes3 = self.conv3_(tes3)
#         res4 = self.conv4(res4)
#         tes4 = self.conv4_(tes4)

#         res2 = F.interpolate(res2, size=(res1h, res1w), mode='bicubic', align_corners=False)
#         res3 = F.interpolate(res3, size=(res1h, res1w), mode='bicubic', align_corners=False)
#         res4 = F.interpolate(res4, size=(res1h, res1w), mode='bicubic', align_corners=False)
#         mid_res = torch.cat([res2, res3, res4], dim=1)

#         tes2 = F.interpolate(tes2, size=(res1h, res1w), mode='bicubic', align_corners=False)
#         tes3 = F.interpolate(tes3, size=(res1h, res1w), mode='bicubic', align_corners=False)
#         tes4 = F.interpolate(tes4, size=(res1h, res1w), mode='bicubic', align_corners=False)
#         mid_tes = torch.cat([tes2, tes3, tes4], dim=1)#1,384,96,96
        
#         return res1, tes1, mid_res, mid_tes








import torch
import torch.nn as nn
import torch.nn.functional as F

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

class CAFA(nn.Module):
    def __init__(self, in_channels=1024, num_heads=4, decode_channels=96,decode_channels1=32):
        super(CAFA, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        
        self.conv1 = BasicConv(128, decode_channels, kernel_size=3, dilation=1, padding=1, groups=decode_channels1, bias=False,
                      relu=False)
        self.conv2 = BasicConv(256, decode_channels, kernel_size=3, dilation=1, padding=1, groups=decode_channels1, bias=False,
                      relu=False)
        self.conv3 = BasicConv(512, decode_channels, kernel_size=3, dilation=1, padding=1, groups=decode_channels1, bias=False,
                      relu=False)
        self.conv4 = BasicConv(1024, decode_channels, kernel_size=3, dilation=1, padding=1, groups=decode_channels1, bias=False,
                      relu=False)

        self.conv1_ = BasicConv(128, decode_channels, kernel_size=3, dilation=1, padding=1, groups=decode_channels1, bias=False,
                      relu=False)
        self.conv2_ = BasicConv(256, decode_channels, kernel_size=3, dilation=1, padding=1, groups=decode_channels1, bias=False,
                      relu=False)
        self.conv3_ = BasicConv(512, decode_channels, kernel_size=3, dilation=1, padding=1, groups=decode_channels1, bias=False,
                      relu=False)
        self.conv4_ = BasicConv(1024, decode_channels, kernel_size=3, dilation=1, padding=1, groups=decode_channels1, bias=False,
                      relu=False)

    def forward(self, res1, res2, res3, res4, tes1, tes2, tes3, tes4):
        res1h, res1w = res1.size()[-2:]  # 96,96

        res1 = self.conv1(res1)
        tes1 = self.conv1_(tes1)
        res2 = self.conv2(res2)
        tes2 = self.conv2_(tes2)
        res3 = self.conv3(res3)
        tes3 = self.conv3_(tes3)
        res4 = self.conv4(res4)
        tes4 = self.conv4_(tes4)

        res2 = F.interpolate(res2, size=(res1h, res1w), mode='bicubic', align_corners=False)
        res3 = F.interpolate(res3, size=(res1h, res1w), mode='bicubic', align_corners=False)
        res4 = F.interpolate(res4, size=(res1h, res1w), mode='bicubic', align_corners=False)
        mid_res = torch.cat([res2, res3, res4], dim=1)

        tes2 = F.interpolate(tes2, size=(res1h, res1w), mode='bicubic', align_corners=False)
        tes3 = F.interpolate(tes3, size=(res1h, res1w), mode='bicubic', align_corners=False)
        tes4 = F.interpolate(tes4, size=(res1h, res1w), mode='bicubic', align_corners=False)
        mid_tes = torch.cat([tes2, tes3, tes4], dim=1)
        
        return res1, tes1, mid_res, mid_tes
