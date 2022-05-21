import torch
import torch.nn.functional as F
from torch import nn

from .blocks import Conv, C3
from .seg_block import PPM


class PSPHead(nn.Module):
    def __init__(self, cins, ncls, strides, ratio):
        super(PSPHead, self).__init__()
        smin = min(strides)
        self.feature_fuse = nn.ModuleList(
            PSPHead._build_fuse(cin, int(cin * ratio), s // smin) for cin, s in zip(cins, strides))  # 融合多个特征层
        c = sum(int(cin * ratio) for cin in cins)
        self.psp = PSPHead._build_out(c, ncls)
        self.conv = nn.Conv2d(ncls, ncls, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.smin = smin

    def forward(self, xs):
        y = torch.cat([m(x) for m, x in zip(self.feature_fuse, xs)], dim=1)  # 特征融合
        y = self.psp(y)
        y = F.interpolate(y, scale_factor=self.smin, mode="bilinear", align_corners=True)
        return self.conv(y)

    @staticmethod
    def _build_fuse(c1, c2, s):
        conv = Conv(c1, c2)
        ups = nn.Upsample(scale_factor=s, mode="bilinear", align_corners=True)
        return nn.Sequential(conv, ups)

    @staticmethod
    def _build_out(c1, c2):
        ppm = PPM(c1)
        c3 = C3(ppm.cout, c2, n=1)
        conv = nn.Conv2d(c2, c2, kernel_size=(1, 1))
        return nn.Sequential(ppm, c3, conv)

