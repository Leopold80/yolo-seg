import copy
from pathlib import Path

import yaml

from .blocks import *
from .head import *


def _make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


def parse_model(d, ch):  # model_dict, input_channels(3)
    nc, gd, gw = d['nc'], d['depth_multiple'], d['width_multiple']

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in (Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus,
                 BottleneckCSP, C3, C3TR, C3SPP, C3Ghost):
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if not output
                c2 = _make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3TR, C3Ghost]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        # elif m is Detect:
        # args.append([ch[x] for x in f])
        # if isinstance(args[1], int):  # number of anchors
        #     args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is PSPHead:
            args.insert(0, [ch[x] for x in f])
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        num_param = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, num_param  # attach index, 'from' index, type, number params
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


class Model(nn.Module):
    def __init__(self, cfg="./model/cfg/nano.yaml"):
        super(Model, self).__init__()
        self.cfg = yaml.safe_load(Path(cfg).open().read())
        self.model, self.save = parse_model(copy.deepcopy(self.cfg), [3])
        self.initialize_weights()

    def forward(self, x):
        y = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x

    def initialize_weights(self):
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True
