import torch
import torch.nn as nn

class ConditionalInstanceNorm2dPlus(nn.Module):
    def __init__(self, num_features, num_classes, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)
        if bias:
            self.embed = nn.Embedding(num_classes, num_features * 3)
            self.embed.weight.data[:, :2 * num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
            self.embed.weight.data[:, 2 * num_features:].zero_()  # Initialise bias at 0
        else:
            self.embed = nn.Embedding(num_classes, 2 * num_features)
            self.embed.weight.data.normal_(1, 0.02)

    def forward(self, x, y):
        means = torch.mean(x, dim=(2, 3))
        m = torch.mean(means, dim=-1, keepdim=True)
        v = torch.var(means, dim=-1, keepdim=True)
        means = (means - m) / (torch.sqrt(v + 1e-5))
        h = self.instance_norm(x)

        if self.bias:
            gamma, alpha, beta = self.embed(y).chunk(3, dim=-1)
            h = h + means[..., None, None] * alpha[..., None, None]
            out = gamma.view(-1, self.num_features, 1, 1) * h + beta.view(-1, self.num_features, 1, 1)
        else:
            gamma, alpha = self.embed(y).chunk(2, dim=-1)
            h = h + means[..., None, None] * alpha[..., None, None]
            out = gamma.view(-1, self.num_features, 1, 1) * h
        return out  

class CondResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, num_levels, normalization) -> None:
        super().__init__()
        self.act = nn.ELU()
        self.normalization = normalization
        self.norm1 = normalization(in_channel, num_levels)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size = 3, padding = 1)
        self.norm2 = normalization(out_channel, num_levels)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size = 3, padding = 1)
        self.shortcut = nn.Conv2d(in_channel, out_channel, 1) if in_channel != out_channel else nn.Identity()

    def forward(self, data, noise_labels):
        shortcut = self.shortcut(data)
        data = self.norm1(data, noise_labels)
        data = self.act(data)
        data = self.conv1(data)
        data = self.norm2(data, noise_labels)
        data = self.act(data)
        data = self.conv2(data)
        return data + shortcut

class CondRes(nn.Module):
    def __init__(self, in_channel, num_levels) -> None:
        super().__init__()
        base_channel = 16
        normalization = ConditionalInstanceNorm2dPlus
        self.conv1 = nn.Conv2d(in_channel, base_channel, 3, padding = 1)
        self.res1 = nn.ModuleList([CondResBlock(base_channel, base_channel, num_levels, normalization),
        CondResBlock(base_channel, 2 * base_channel, num_levels, normalization)])
        self.res2 = nn.ModuleList([CondResBlock(2 * base_channel, 2 * base_channel, num_levels, normalization),
        CondResBlock(2 * base_channel, 2 * base_channel, num_levels, normalization)])
        self.res3 = nn.ModuleList([CondResBlock(2 * base_channel, 4 * base_channel, num_levels, normalization),
        CondResBlock(4 * base_channel, 4 * base_channel, num_levels, normalization)])
        self.conv2 = nn.Conv2d(4 * base_channel, in_channel, 3, padding = 1)

    def cond_module_forward(self, module, data, noise_labels):
        for m in module:
            data = m(data, noise_labels)
        return data

    def forward(self, data, noise_labels):
        data = 2 * data - 1.
        h = self.conv1(data)
        h1 = self.cond_module_forward(self.res1, h, noise_labels)
        h2 = self.cond_module_forward(self.res2, h1, noise_labels)
        h3 = self.cond_module_forward(self.res3, h2, noise_labels)
        out = self.conv2(h3)
        return out