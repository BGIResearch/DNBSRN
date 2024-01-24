import torch.nn as nn


# Shallow Residual Block (SRB)
class SRB(nn.Module):
    def __init__(self, n_feat, kernel_size):
        super(SRB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size, padding=(kernel_size//2)),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return x + self.body(x)


# DNB Super-Resolution Network (DNBSRN)
class DNBSRN(nn.Module):
    def __init__(self, num_channels=1, num_branches=3, n_feat=48, kernel_size=7):
        super(DNBSRN, self).__init__()
        self.num_branches = num_branches
        self.head = nn.Conv2d(num_channels, n_feat, kernel_size, padding=(kernel_size//2))
        self.body = nn.ModuleList([SRB(n_feat, kernel_size) for _ in range(self.num_branches)])
        self.conv = nn.ModuleList(
            [nn.Conv2d(n_feat, num_channels, kernel_size, padding=(kernel_size//2)) for _ in range(self.num_branches)])

    def forward(self, x):
        x0 = self.head(x)
        for i in range(self.num_branches):
            x0 = self.body[i](x0)
            x = x + self.conv[i](x0)
        return x
