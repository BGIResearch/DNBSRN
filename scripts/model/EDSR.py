import torch.nn as nn


# Residual Block (RB)
class RB(nn.Module):
    def __init__(self, n_feats, kernel_size, res_scale):
        super(RB, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size//2)),
                                  nn.ReLU(True),
                                  nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size//2)))
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


# Enhanced Deep Super-Resolution network (EDSR)
class EDSR(nn.Module):
    def __init__(self, n_resblocks=32, n_feats=256, kernel_size=3, res_scale=0.1):
        super(EDSR, self).__init__()
        # define head module
        m_head = [nn.Conv2d(1, n_feats, kernel_size, padding=(kernel_size//2))]
        # define body module
        m_body = [RB(n_feats, kernel_size, res_scale) for _ in range(n_resblocks)]
        m_body.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size//2)))
        # define tail module
        m_tail = [nn.Conv2d(n_feats, 1, kernel_size, padding=(kernel_size//2))]
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x
