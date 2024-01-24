import torch.nn as nn
import torch.nn.functional as F


# modification of Enhanced Spatial Attention (ESA)
class ESA(nn.Module):
    def __init__(self, esa_channels, n_feats):
        super(ESA, self).__init__()
        f = esa_channels
        self.conv1 = nn.Conv2d(n_feats, f, kernel_size=1)
        self.conv_f = nn.Conv2d(f, f, kernel_size=1)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


# Residual Local Feature Block (RLFB)
class RLFB(nn.Module):
    def __init__(self, in_channels, esa_channels=16):
        super(RLFB, self).__init__()
        self.c1_r = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.c2_r = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.c3_r = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.c5 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.esa = ESA(esa_channels, in_channels)
        self.act = nn.LeakyReLU(negative_slope=0.05, inplace=True)

    def forward(self, x):
        out = (self.c1_r(x))
        out = self.act(out)
        out = (self.c2_r(out))
        out = self.act(out)
        out = (self.c3_r(out))
        out = self.act(out)
        out = out + x
        out = self.esa(self.c5(out))
        return out


# Residual Local Feature Network (RLFN)
class RLFN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, feature_channels=52):
        super(RLFN, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1)
        self.block_1 = RLFB(feature_channels)
        self.block_2 = RLFB(feature_channels)
        self.block_3 = RLFB(feature_channels)
        self.block_4 = RLFB(feature_channels)
        self.block_5 = RLFB(feature_channels)
        self.block_6 = RLFB(feature_channels)
        self.conv_2 = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1)
        self.upsampler = nn.Conv2d(feature_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        out_feature = self.conv_1(x)
        out_b1 = self.block_1(out_feature)
        out_b2 = self.block_2(out_b1)
        out_b3 = self.block_3(out_b2)
        out_b4 = self.block_4(out_b3)
        out_b5 = self.block_5(out_b4)
        out_b6 = self.block_6(out_b5)
        out_low_resolution = self.conv_2(out_b6) + out_feature
        output = self.upsampler(out_low_resolution)
        return output
