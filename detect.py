from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from model import Model, letterbox


def img_generator(inputsz=(640, 640), root="./datasets/dataset2/jpg"):
    for im0 in Path(root).glob("*.jpg"):
        im0 = cv2.imread(str(im0))
        # im0 = cv2.resize(im0, inputsz)
        im0, r = letterbox(im0, inputsz)
        im = im0.transpose((2, 0, 1))
        yield torch.from_numpy(im).float().unsqueeze(0), im0


@torch.no_grad()
def detect():
    m = Model(cfg="./model/cfg/nano-p2.yaml")
    m.load_state_dict(torch.load(Path("./runs/last-p2.pt")), strict=True)
    for x, im in img_generator():
        y = m(x)
        y = F.softmax(y, dim=1).squeeze(0)
        mask = torch.argmax(y, dim=0).unsqueeze(-1)

        mask = mask.numpy().repeat(3, axis=2).astype(np.uint8)
        mask = mask * np.array([0, 255, 255], dtype=np.uint8).reshape((1, 1, 3))
        im = cv2.addWeighted(im, 0.5, mask, 0.5, gamma=1)

        cv2.imshow("im", im)
        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detect()
