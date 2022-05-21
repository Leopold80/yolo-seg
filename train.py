import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from augment import Compose, LetterBox, RandomPerspective
from dataloader import SEGDataset
from model import Model


@logger.catch()
def train():
    dev = torch.device("cuda")

    epoch = 100
    size = (640, 640)
    batchsize = 1

    savedir = Path("./runs")
    savedir.mkdir(parents=True, exist_ok=True)
    last, best = savedir / "last-p2.pt", savedir / "best.pt"

    m = Model(cfg="./model/cfg/nano-p2.yaml")
    max_stride = 32
    m.to(dev)
    dataset = SEGDataset(nc=2, aug=Compose([RandomPerspective(), LetterBox(size)]))
    loader = DataLoader(dataset, batchsize, True)

    opt = torch.optim.Adam(m.parameters(), 0.01, (0.9, 0.999))
    bce_loss = nn.BCEWithLogitsLoss()

    for e in range(epoch):
        m.train()
        with tqdm(loader) as pbar:
            pbar.set_description("epoch: {}/{}".format(e + 1, epoch))
            loss_sum = 0
            n = len(pbar)
            for img, label in pbar:
                opt.zero_grad()
                img = img.to(dev)
                label = label.to(dev)
                img = F.interpolate(img, size, mode="bilinear", align_corners=True)
                label = F.interpolate(label, size, mode="nearest")
                pred = m.forward(img)
                loss = bce_loss.forward(pred, label)
                loss.backward()
                opt.step()
                loss_ = loss.detach().item()
                pbar.set_postfix_str(
                    "loss: {:.4f}, gpu_mem: {:.2f}G".format(loss_, torch.cuda.memory_reserved() / 1E9)
                )
                loss_sum += loss_

            logger.info("avg loss: {:.5f}".format(loss_sum / n))
            torch.save(m.state_dict(), str(last))


train()
