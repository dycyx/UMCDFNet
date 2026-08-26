import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import math
import torch.nn.functional as F

class Bcnn_AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        
        self.conv_spatial = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=9, groups=dim, dilation=3)
        
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.Sigmoid() 

    def forward(self, x):
        u = x
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        attn = self.act(attn)
        return u * attn

class From_Channel(nn.Module):
    def __init__(self, dim, reduction=4):
        super(From_Channel, self).__init__()
        in_channels = dim * 2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, main_feat, aux_feat):
        B, C, H, W = main_feat.shape
        x = torch.cat((main_feat, aux_feat), dim=1) 
        
        avg_v = self.avg_pool(x).view(B, -1)
        max_v = self.max_pool(x).view(B, -1)
        
        avg_se = self.mlp(avg_v).view(B, -1, 1, 1)
        max_se = self.mlp(max_v).view(B, -1, 1, 1)
        
        channel_weights = self.sigmoid(avg_se + max_se)
        return channel_weights[:, :C, :, :]

class From_Spatial(nn.Module):
    def __init__(self, in_channels, reduction=4, kernel_size=1):
        super(From_Spatial, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(4, 4 * reduction, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * reduction, 1, kernel_size=kernel_size, padding=kernel_size//2), 
            nn.Sigmoid()
        )

    def forward(self, main_feat, aux_feat):
        B, C, H, W = main_feat.shape
        
        main_mean = torch.mean(main_feat, dim=1, keepdim=True)
        main_max, _ = torch.max(main_feat, dim=1, keepdim=True)
        
        aux_mean = torch.mean(aux_feat, dim=1, keepdim=True)
        aux_max, _ = torch.max(aux_feat, dim=1, keepdim=True)
        
        x_cat = torch.cat((main_mean, main_max, aux_mean, aux_max), dim=1) # [B, 4, H, W]
        
        spatial_weights = self.mlp(x_cat) # [B, 1, H, W]
        return spatial_weights

class SDFT(nn.Module):
    def __init__(self, in_dim=384, out_dim=256, reduction=4):
        super(SDFT, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.att_init_rgb = Bcnn_AttentionModule(in_dim)
        self.att_init_t = Bcnn_AttentionModule(in_dim)
        
        self.ch_rgb = From_Channel(in_dim, reduction=reduction)
        self.sp_rgb = From_Spatial(in_dim, reduction=reduction)
        
        self.ch_t = From_Channel(in_dim, reduction=reduction)
        self.sp_t = From_Spatial(in_dim, reduction=reduction)
        self.norm = nn.BatchNorm2d(in_dim * 2)
        self.ffn = nn.Sequential(
            nn.Conv2d(in_dim * 2, in_dim * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_dim * 2, in_dim * 2, kernel_size=1)
        )
        
        self.att_final = Bcnn_AttentionModule(in_dim * 2) 
        
        self.project_out = nn.Conv2d(in_dim * 2, out_dim * 2, kernel_size=1)

    def forward(self, rgb, t):
        B, C, H, W = rgb.shape
        
        rgb_att = self.att_init_rgb(rgb)
        t_att = self.att_init_t(t)

        w_ch_rgb = self.ch_rgb(rgb_att, t_att)
        w_sp_rgb = self.sp_rgb(rgb_att, t_att) 
        w_total_rgb = w_ch_rgb * w_sp_rgb
        out_rgb = rgb_att + rgb_att * w_total_rgb
        
        w_ch_t = self.ch_t(t_att, rgb_att)
        w_sp_t = self.sp_t(t_att, rgb_att)
        w_total_t = w_ch_t * w_sp_t
        out_t = t_att + t_att * w_total_t
        
        fused_cat = torch.cat([out_rgb, out_t], dim=1)
        ffn_out = self.ffn(self.norm(fused_cat))
        fused_res = fused_cat + ffn_out 
        final_att = self.att_final(fused_res)
        
        output = self.project_out(final_att)
        
        return output


# import torch
# import torch.nn as nn
# from timm.models.layers import trunc_normal_
# import math
# import torch.nn.functional as F

# class SDFT(nn.Module):
#     def __init__(self, in_dim=288, out_dim=96, reduction=4):
#         super(SDFT, self).__init__()
        
#         self.project_out = nn.Conv2d(in_dim * 2, out_dim * 2, kernel_size=1)

#     def forward(self, rgb, t):
        
#         fused_cat = torch.cat([rgb, t], dim=1)
#         output = self.project_out(fused_cat)
        
#         return output