import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, normalization, dilation = 1) -> None:
        super().__init__()
        self.act = nn.ELU()
        self.normalization = normalization
        self.norm1 = normalization(in_channel)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size = 3, padding = dilation, dilation = dilation)
        self.norm2 = normalization(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size = 3, padding = dilation, dilation = dilation)
        self.shortcut = nn.Conv2d(in_channel, out_channel, 1, dilation = dilation) if in_channel != out_channel else nn.Identity()

    def forward(self, data):
        shortcut = self.shortcut(data)
        data = self.norm1(data)
        data = self.act(data)
        data = self.conv1(data)
        data = self.norm2(data)
        data = self.act(data)
        data = self.conv2(data)
        return data + shortcut

class ResNet(nn.Module):
    def __init__(self, in_channel, sigmas) -> None:
        super().__init__()
        base_channel = 16
        self.sigmas = nn.Parameter(sigmas, requires_grad = False)
        normalization = nn.InstanceNorm2d
        self.act = nn.ELU()
        self.norm = normalization(4 * base_channel)
        self.conv1 = nn.Conv2d(in_channel, base_channel, 3, padding = 1)
        # self.down = 
        self.res1 = nn.ModuleList([
            ResBlock(base_channel, base_channel, normalization),
            ResBlock(base_channel, 2 * base_channel, normalization),
            ])
        self.res2 = nn.ModuleList([
            ResBlock(2 * base_channel, 2 * base_channel, normalization),
            ResBlock(2 * base_channel, 4 * base_channel, normalization)])
        self.res3 = nn.ModuleList([
            ResBlock(4 * base_channel, 4 * base_channel, normalization, dilation=2),
            ResBlock(4 * base_channel, 4 * base_channel, normalization, dilation=2)])
        self.res4 = nn.ModuleList([
            ResBlock(4 * base_channel, 4 * base_channel, normalization, dilation=4),
            ResBlock(4 * base_channel, 4 * base_channel, normalization, dilation=4)
        ])
        # self.up = 
        self.conv2 = nn.Conv2d(4 * base_channel, in_channel, 3, padding = 1)

    def module_forward(self, module, data):
        for m in module:
            data = m(data)
        return data

    def forward(self, data):
        data = 2 * data - 1.    #convert to [-1,1] range
        h = self.conv1(data)
        h1 = self.module_forward(self.res1, h)
        h2 = self.module_forward(self.res2, h1)
        h3 = self.module_forward(self.res3, h2)
        h4 = self.module_forward(self.res4, h3)
        out = self.norm(h4)
        out = self.act(out)
        out = self.conv2(out)
        assert out.shape == data.shape
        return out
