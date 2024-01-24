import torch
import torch.nn as nn
import torch.nn.functional as F


# Enhanced Spatial Attention (ESA)
class ESA(nn.Module):
    def __init__(self, n_feats):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = nn.Conv2d(n_feats, f, kernel_size=1)
        self.conv_f = nn.Conv2d(f, f, kernel_size=1)
        self.conv_max = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv3_ = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


# Residual Feature Distillation Block (RFDB)
class RFDB(nn.Module):
    def __init__(self, in_channels):
        super(RFDB, self).__init__()
        self.dc = self.distilled_channels = in_channels//2
        self.rc = self.remaining_channels = in_channels
        self.c1_d = nn.Conv2d(in_channels, self.dc, kernel_size=1, padding=0)
        self.c1_r = nn.Conv2d(in_channels, self.rc, kernel_size=3, padding=1)
        self.c2_d = nn.Conv2d(self.remaining_channels, self.dc, kernel_size=1, padding=0)
        self.c2_r = nn.Conv2d(self.remaining_channels, self.rc, kernel_size=3, padding=1)
        self.c3_d = nn.Conv2d(self.remaining_channels, self.dc, kernel_size=1, padding=0)
        self.c3_r = nn.Conv2d(self.remaining_channels, self.rc, kernel_size=3, padding=1)
        self.c4 = nn.Conv2d(self.remaining_channels, self.dc, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(negative_slope=0.05, inplace=True)
        self.c5 = nn.Conv2d(self.dc*4, in_channels, kernel_size=1, padding=0)
        self.esa = ESA(in_channels)

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1+input)
        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2+r_c1)
        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3+r_c2)
        r_c4 = self.act(self.c4(r_c3))
        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.esa(self.c5(out))
        return out_fused


# enhanced Residual Feature Distillation Network (RFDN)
class RFDN(nn.Module):
    def __init__(self, in_nc=1, nf=50, num_modules=4, out_nc=1):
        super(RFDN, self).__init__()
        self.fea_conv = nn.Conv2d(in_nc, nf, kernel_size=3, padding=1)
        self.B1 = RFDB(in_channels=nf)
        self.B2 = RFDB(in_channels=nf)
        self.B3 = RFDB(in_channels=nf)
        self.B4 = RFDB(in_channels=nf)
        self.c = nn.Sequential(nn.Conv2d(nf*num_modules, nf, kernel_size=1, padding=0),
                               nn.LeakyReLU(negative_slope=0.05, inplace=True))
        self.LR_conv = nn.Conv2d(nf, nf, kernel_size=3, padding=1)
        self.SRconv = nn.Conv2d(nf, out_nc, kernel_size=3, padding=1)

    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.LR_conv(out_B)+out_fea
        output = self.SRconv(out_lr)
        return output
