import math

import torch
import torch.nn.functional


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / (fan_out // m.groups)))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, torch.nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.uniform_(-1.0 / math.sqrt(m.weight.size()
                                                    [0]), 1.0 / math.sqrt(m.weight.size()[0]))
            m.bias.data.zero_()


class SiLU(torch.nn.Module):
    """
    Sigmoid-weighted Linear Unit
    https://arxiv.org/pdf/1710.05941.pdf
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.mul_(x.sigmoid()) if self.inplace else x.mul(x.sigmoid())


class Conv2d(torch.nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=False):
        super().__init__(in_channels, out_channels,
                         kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        s = self.stride
        d = self.dilation
        k = self.weight.shape[-2:]
        h, w = x.size()[-2:]
        pad_h = max((math.ceil(h / s[0]) - 1) *
                    s[0] + (k[0] - 1) * d[0] + 1 - h, 0)
        pad_w = max((math.ceil(w / s[1]) - 1) *
                    s[1] + (k[1] - 1) * d[1] + 1 - w, 0)
        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=0)

        return torch.nn.functional.conv2d(x, self.weight, self.bias, self.stride, (0, 0), self.dilation, self.groups)


class Conv(torch.nn.Module):
    def __init__(self, args, in_ch, out_ch, activation, k=1, s=1, g=1):
        super().__init__()
        if args:
            self.conv = Conv2d(in_ch, out_ch, k, s, 1, g, bias=False)
        else:
            self.conv = torch.nn.Conv2d(
                in_ch, out_ch, k, s, k // 2, 1, g, bias=False)
        self.norm = torch.nn.BatchNorm2d(out_ch, 0.001, 0.01)
        self.silu = activation

    def forward(self, x):
        return self.silu(self.norm(self.conv(x)))


class SE(torch.nn.Module):
    """
    Squeeze-and-Excitation
    https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(self, ch, r):
        super().__init__()
        self.se = torch.nn.Sequential(torch.nn.Conv2d(ch, ch // (4 * r), 1),
                                      torch.nn.SiLU(),
                                      torch.nn.Conv2d(ch // (4 * r), ch, 1),
                                      torch.nn.Sigmoid())

    def forward(self, x):
        return x * self.se(x.mean((2, 3), keepdim=True))


class Residual(torch.nn.Module):
    """
    Inverted Residuals and Linear Bottlenecks
    https://arxiv.org/pdf/1801.04381.pdf
    """

    def __init__(self, args, in_ch, out_ch, s, r, fused=True):
        super().__init__()
        identity = torch.nn.Identity()
        if fused:
            if args and r == 1:
                features = [Conv(args, in_ch, r * in_ch,
                                 torch.nn.SiLU(), 3, s)]
            else:
                features = [Conv(args, in_ch, r * in_ch, torch.nn.SiLU(), 3, s),
                            Conv(args, r * in_ch, out_ch, identity)]
        else:
            if r == 1:
                features = [Conv(args, r * in_ch, r * in_ch, torch.nn.SiLU(), 3, s, r * in_ch),
                            SE(r * in_ch, r),
                            Conv(args, r * in_ch, out_ch, identity)]
            else:
                features = [Conv(args, in_ch, r * in_ch, torch.nn.SiLU()),
                            Conv(args, r * in_ch, r * in_ch,
                                 torch.nn.SiLU(), 3, s, r * in_ch),
                            SE(r * in_ch, r),
                            Conv(args, r * in_ch, out_ch, identity)]
        self.add = s == 1 and in_ch == out_ch
        self.res = torch.nn.Sequential(*features)

    def forward(self, x):
        return x + self.res(x) if self.add else self.res(x)
