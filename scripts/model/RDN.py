import torch
from torch import nn


# Dense Layer (DL)
class DL(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DL, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


# Residual Dense Block (RDB)
class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DL(in_channels+growth_rate*i, growth_rate) for i in range(num_layers)])
        self.lff = nn.Conv2d(in_channels+growth_rate*num_layers, growth_rate, kernel_size=1)

    def forward(self, x):
        return x + self.lff(self.layers(x))


# Residual Dense Network (RDN)
class RDN(nn.Module):
    def __init__(self, num_channels=1, num_features=64, growth_rate=64, num_blocks=16, num_layers=8):
        super(RDN, self).__init__()
        self.D = num_blocks
        self.sfe1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=1)
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.rdbs = nn.ModuleList([RDB(num_features, growth_rate, num_layers)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(growth_rate, growth_rate, num_layers))
        self.gff = nn.Sequential(
            nn.Conv2d(growth_rate*self.D, num_features, kernel_size=1),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
        self.output = nn.Conv2d(num_features, num_channels, kernel_size=3, padding=1)

    def forward(self, x):
        sfe1 = self.sfe1(x)
        x = self.sfe2(sfe1)
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)
        x = self.gff(torch.cat(local_features, 1)) + sfe1
        x = self.output(x)
        return x
