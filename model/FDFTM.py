# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from pytorch_wavelets import DWTForward

# class FDFT(nn.Module):
#     def __init__(self, channels):
#         super(FDFT, self).__init__()
        
#         self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.wt = DWTForward(J=1, mode='zero', wave='haar')
        
#         self.conv_bn_relu_L = nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(channels),
#             nn.ReLU(inplace=True),
#         )
#         self.conv_bn_relu_H = nn.Sequential(
#             nn.Conv2d(channels * 3, channels, kernel_size=3, padding=1), 
#             nn.BatchNorm2d(channels),
#             nn.ReLU(inplace=True),
#         )
        
#         self.L_fuse_net = nn.Sequential(
#             nn.Conv2d(channels * 2, channels, kernel_size=1),
#             nn.BatchNorm2d(channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels, channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(channels),
#             nn.ReLU(inplace=True)
#         )
#         self.H_attention = nn.Sequential(
#             nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels, 2, kernel_size=1) # 输出 2 个通道用于 softmax
#         )
        
#         self.H_fuse_net = nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels, channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(channels),
#             nn.ReLU(inplace=True)
#         )
        
        
#         self.out_conv_L = nn.Sequential(
#             nn.Conv2d(channels, channels//3, kernel_size=1),
#             nn.BatchNorm2d(channels//3),
#             nn.ReLU(inplace=True),
#         )
#         self.out_conv_H = nn.Sequential(
#             nn.Conv2d(channels, channels//3, kernel_size=1),
#             nn.BatchNorm2d(channels//3),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, rgb, t):

#         f_RL, f_RH = self.wt(self.up2(rgb))
#         f_RH_tensor = f_RH[0] 
#         f_RH = torch.cat([f_RH_tensor[:, :, 0, :, :], 
#                           f_RH_tensor[:, :, 1, :, :], 
#                           f_RH_tensor[:, :, 2, :, :]], dim=1) 
#         f_RH = self.conv_bn_relu_H(f_RH)
#         f_RL = self.conv_bn_relu_L(f_RL)

#         f_TL, f_TH = self.wt(self.up2(t))
#         f_TH_tensor = f_TH[0]
#         f_TH = torch.cat([f_TH_tensor[:, :, 0, :, :], 
#                           f_TH_tensor[:, :, 1, :, :], 
#                           f_TH_tensor[:, :, 2, :, :]], dim=1)
#         f_TH = self.conv_bn_relu_H(f_TH)
#         f_TL = self.conv_bn_relu_L(f_TL)

#         cat_L = torch.cat([f_RL, f_TL], dim=1)
#         L_fused = self.L_fuse_net(cat_L) + (f_RL + f_TL) 

#         cat_H = torch.cat([f_RH, f_TH], dim=1)
#         h_weights = self.H_attention(cat_H)
#         h_weights = F.softmax(h_weights, dim=1) 
#         w_rgb = h_weights[:, 0:1, :, :]
#         w_t = h_weights[:, 1:2, :, :]
#         H_weighted = f_RH * w_rgb + f_TH * w_t
#         H_fused = self.H_fuse_net(H_weighted) + H_weighted

#         out_l = self.out_conv_L(L_fused)
#         out_h = self.out_conv_H(H_fused)

#         return out_l, out_h






# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from pytorch_wavelets import DWTForward

# class FDFT(nn.Module):
#     def __init__(self, channels):
#         super(FDFT, self).__init__()

#         self.FUSE = nn.Sequential(
#             nn.Conv2d(channels * 2, channels, kernel_size=1),
#             nn.BatchNorm2d(channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels, channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(channels),
#             nn.ReLU(inplace=True)
#         )
        
#         self.out_conv_L = nn.Sequential(
#             nn.Conv2d(channels, channels//3, kernel_size=1),
#             nn.BatchNorm2d(channels//3),
#             nn.ReLU(inplace=True),
#         )
#         self.out_conv_H = nn.Sequential(
#             nn.Conv2d(channels, channels//3, kernel_size=1),
#             nn.BatchNorm2d(channels//3),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, rgb, t):

#         fused = self.FUSE(torch.cat([rgb, t], dim=1))

#         out_l = self.out_conv_L(fused)
#         out_h = self.out_conv_H(fused)

#         return out_l, out_h







import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward

class FDFT(nn.Module):
    def __init__(self, channels, groups=8):
        super(FDFT, self).__init__()
        self.channels = channels
        self.groups = groups   # 分组卷积的分组数，所有卷积均使用此值（除特例外）

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.wt = DWTForward(J=1, mode='zero', wave='haar')

        self.conv_bn_relu_L = nn.Sequential(
            # Depthwise 3x3
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.conv_bn_relu_H = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=3, padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.L_fuse_net = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            # Depthwise 3x3
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            # Pointwise 1x1
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.H_attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, groups=groups, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        )

        self.H_fuse_net = nn.Sequential(
            # 第一个 depthwise separable
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.out_conv_L = nn.Sequential(
            nn.Conv2d(channels, channels // 3, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(channels // 3),
            nn.ReLU(inplace=True),
        )
        self.out_conv_H = nn.Sequential(
            nn.Conv2d(channels, channels // 3, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(channels // 3),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb, t):
        f_RL, f_RH = self.wt(self.up2(rgb))
        f_RH_tensor = f_RH[0]  # shape: [B, C, 3, H/2, W/2]
        f_RH = torch.cat([f_RH_tensor[:, :, 0, :, :],
                          f_RH_tensor[:, :, 1, :, :],
                          f_RH_tensor[:, :, 2, :, :]], dim=1)  # [B, 3C, H/2, W/2]
        f_RH = self.conv_bn_relu_H(f_RH)
        f_RL = self.conv_bn_relu_L(f_RL)

        f_TL, f_TH = self.wt(self.up2(t))
        f_TH_tensor = f_TH[0]
        f_TH = torch.cat([f_TH_tensor[:, :, 0, :, :],
                          f_TH_tensor[:, :, 1, :, :],
                          f_TH_tensor[:, :, 2, :, :]], dim=1)
        f_TH = self.conv_bn_relu_H(f_TH)
        f_TL = self.conv_bn_relu_L(f_TL)

        cat_L = torch.cat([f_RL, f_TL], dim=1)
        L_fused = self.L_fuse_net(cat_L) + (f_RL + f_TL)

        cat_H = torch.cat([f_RH, f_TH], dim=1)
        h_weights = self.H_attention(cat_H)          # [B, 2, H, W]
        h_weights = F.softmax(h_weights, dim=1)      # 沿通道做 softmax
        w_rgb = h_weights[:, 0:1, :, :]
        w_t = h_weights[:, 1:2, :, :]
        H_weighted = f_RH * w_rgb + f_TH * w_t
        H_fused = self.H_fuse_net(H_weighted) + H_weighted

        out_l = self.out_conv_L(L_fused)
        out_h = self.out_conv_H(H_fused)

        return out_l, out_h