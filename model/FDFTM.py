import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward

class FDFT(nn.Module):
    def __init__(self, channels):
        super(FDFT, self).__init__()
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        
        self.conv_bn_relu_L = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.conv_bn_relu_H = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=3, padding=1), 
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        
        self.L_fuse_net = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.H_attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2, kernel_size=1) # 输出 2 个通道用于 softmax
        )
        
        self.H_fuse_net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        
        self.out_conv_L = nn.Sequential(
            nn.Conv2d(channels, channels//3, kernel_size=1),
            nn.BatchNorm2d(channels//3),
            nn.ReLU(inplace=True),
        )
        self.out_conv_H = nn.Sequential(
            nn.Conv2d(channels, channels//3, kernel_size=1),
            nn.BatchNorm2d(channels//3),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb, t):

        f_RL, f_RH = self.wt(self.up2(rgb))
        f_RH_tensor = f_RH[0] 
        f_RH = torch.cat([f_RH_tensor[:, :, 0, :, :], 
                          f_RH_tensor[:, :, 1, :, :], 
                          f_RH_tensor[:, :, 2, :, :]], dim=1) 
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
        h_weights = self.H_attention(cat_H)
        h_weights = F.softmax(h_weights, dim=1) 
        w_rgb = h_weights[:, 0:1, :, :]
        w_t = h_weights[:, 1:2, :, :]
        H_weighted = f_RH * w_rgb + f_TH * w_t
        H_fused = self.H_fuse_net(H_weighted) + H_weighted

        out_l = self.out_conv_L(L_fused)
        out_h = self.out_conv_H(H_fused)

        return out_l, out_h

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_size = 8
    channels = 384
    h, w = 96, 96
    
    rgb = torch.randn(batch_size, channels, h, w).to(device)
    t = torch.randn(batch_size, channels, h, w).to(device)

    net = FDFT(channels).to(device)
    net.eval() 
    
    with torch.no_grad():
        out_l, out_h = net(rgb, t)
    
    print(f"Input Shape: {rgb.shape}")
    print(f"Output Low Shape:  {out_l.shape}")
    print(f"Output High Shape: {out_h.shape}")
    
    assert out_l.shape == (batch_size, channels, h, w), "Low freq output shape mismatch"
    assert out_h.shape == (batch_size, channels, h, w), "High freq output shape mismatch"
    print("Test Passed!")