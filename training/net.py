import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from voice_rectifier.model import BLOCKS, STEM


class ConvBN(nn.Module):
    def __init__(self, cin, cout, k, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride=stride, padding=k // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.dw = ConvBN(cin, cin, 3, stride, groups=cin)
        self.pw = ConvBN(cin, cout, 1)
        self.residual = stride == 1 and cin == cout

    def forward(self, x):
        y = self.pw(self.dw(x))
        return y + x if self.residual else y


class Net(nn.Module):
    """PyTorch twin of voice_rectifier.model.Model, used for training."""

    def __init__(self):
        super().__init__()
        self.stem = ConvBN(1, STEM, 3, stride=2)
        blocks, cin = [], STEM
        for cout, stride in BLOCKS:
            blocks.append(Block(cin, cout, stride))
            cin = cout
        self.blocks = nn.ModuleList(blocks)
        self.pool_bn = nn.BatchNorm1d(2 * cin)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(2 * cin, 128)

    def forward(self, x):
        x = self.stem(x.unsqueeze(1))
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=2)
        pooled = torch.cat([x.mean(dim=2), torch.sqrt(x.var(dim=2, unbiased=False) + 1e-5)], dim=1)
        return F.normalize(self.fc(self.dropout(self.pool_bn(pooled))), dim=1)


def _fold(conv_bn):
    bn = conv_bn.bn
    scale = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach().double()
    w = conv_bn.conv.weight.detach().double() * scale.view(-1, 1, 1, 1)
    b = bn.bias.detach().double() - bn.running_mean.detach().double() * scale
    return w.numpy(), b.numpy()


def to_numpy_weights(net):
    """Fold batch-norm into the layers and store weights as int8 (one scale per output channel)."""
    net = net.cpu().eval()
    layers = {"stem": _fold(net.stem)}
    for i, block in enumerate(net.blocks):
        layers[f"b{i}.dw"] = _fold(block.dw)
        layers[f"b{i}.pw"] = _fold(block.pw)
    bn = net.pool_bn
    scale = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach().double()
    shift = bn.bias.detach().double() - bn.running_mean.detach().double() * scale
    w, b = net.fc.weight.detach().double(), net.fc.bias.detach().double()
    layers["fc"] = ((w * scale[None]).numpy(), (b + w @ shift).numpy())
    out = {}
    for name, (w, b) in layers.items():
        flat = w.reshape(len(w), -1)
        scale = np.maximum(np.abs(flat).max(axis=1), 1e-12) / 127
        out[name + ".wq"] = np.clip(np.round(flat / scale[:, None]), -127, 127).astype(np.int8).reshape(w.shape)
        out[name + ".ws"] = scale.astype(np.float32)
        out[name + ".b"] = b.astype(np.float32)
    return out
