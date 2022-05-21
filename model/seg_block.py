import torch
import torch.nn.functional as F
from torch import nn

from .blocks import Conv


class PPM(nn.Module):
    def __init__(self, c1, sizes=(2, 3, 6)):
        super(PPM, self).__init__()
        cmid = c1 // len(sizes)
        self.pool = nn.ModuleList(PPM._build_pool(c1, cmid, size) for size in sizes)
        self.cout = cmid * len(sizes) + c1

    def forward(self, x):
        h, w = x.shape[2:]
        ys = (F.interpolate(m(x), (h, w), mode='bilinear', align_corners=True) for m in self.pool)
        return torch.cat([x, *ys], dim=1)

    @staticmethod
    def _build_pool(c1, c2, size):
        pool = nn.AdaptiveAvgPool2d(size)
        conv = Conv(c1, c2)
        return nn.Sequential(pool, conv)
