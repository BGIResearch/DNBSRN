import torch.nn as nn


# Channel Attention (CA)
class CA(nn.Module):
    def __init__(self, channel, reduction):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel//reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel//reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction):
        super(RCAB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size, padding=(kernel_size//2), bias=True),
            nn.ReLU(True),
            nn.Conv2d(n_feat, n_feat, kernel_size, padding=(kernel_size//2), bias=True),
            CA(n_feat, reduction))

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


# Residual Group (RG)
class RG(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, n_resblocks):
        super(RG, self).__init__()
        modules_body = [RCAB(n_feat, kernel_size, reduction) for _ in range(n_resblocks)]
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding=(kernel_size//2), bias=True))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


# Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    def __init__(self, in_nf=1, out_nf=1, n_resgroups=10, n_resblocks=20, n_feats=64, kernel_size=3, reduction=16):
        super(RCAN, self).__init__()
        # define head module
        modules_head = [nn.Conv2d(in_nf, n_feats, kernel_size, padding=(kernel_size//2), bias=True)]
        # define body module
        modules_body = [RG(n_feats, kernel_size, reduction, n_resblocks=n_resblocks) for _ in range(n_resgroups)]
        modules_body.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size//2), bias=True))
        # define tail module
        modules_tail = [nn.Conv2d(n_feats, out_nf, kernel_size, padding=(kernel_size//2), bias=True)]
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x 
