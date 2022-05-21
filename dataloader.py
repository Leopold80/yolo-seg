from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from augment import Compose, LetterBox, RandomPerspective


class SEGDataset(Dataset):
    def __init__(self, nc, aug=Compose(transpose=[RandomPerspective(), LetterBox((640, 640))])):
        super(SEGDataset, self).__init__()
        self.aug = aug
        self.ids = [_id.with_suffix("") for _id in Path("./datasets/dataset2/jpg").glob("*.jpg")]
        self.img = "{}.jpg"
        self.label = "{}.png"
        self.nc = nc

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        # id_ = self.ids[index]
        # img = cv2.imread(str(Path(self.img.format(id_))))
        # label = cv2.imread(str(Path(self.label.format(id_))))
        # img = cv2.resize(img, self.inputsz)
        # label = cv2.resize(label, self.inputsz, cv2.INTER_NEAREST)
        # label = label[:, :, 0] if len(label.shape) == 3 else label
        # onehot_label = np.eye(self.nc)[label.reshape([-1])].reshape(*self.inputsz, self.nc)
        # # for i, im in enumerate(onehot_label.transpose((2, 0, 1))):
        # #     im *= 100
        # #     cv2.imshow("vis{}".format(i), im)
        # # cv2.imshow("img", img)
        # # cv2.waitKey()
        # # cv2.destroyAllWindows()
        #
        # img = img.transpose((2, 0, 1))[::-1, :, :]
        # onehot_label = onehot_label.transpose((2, 0, 1))
        # img = np.ascontiguousarray(img)
        # onehot_label = np.ascontiguousarray(onehot_label)
        # return torch.from_numpy(img).float(), torch.from_numpy(onehot_label).float()

        id_ = self.ids[index]
        img = cv2.imread(str(Path(self.img.format(id_))))
        label = cv2.imread(str(Path(self.label.format(id_))))
        img, label = self.aug.transform(img, label)  # 数据增强

        label = label[:, :, 0] if len(label.shape) == 3 else label
        label = np.eye(self.nc)[label.reshape([-1])].reshape(*label.shape[:2], self.nc)

        # label = (label[:, :, 1:2].repeat(3, axis=2) * np.array([0, 255, 255], dtype=np.uint8).reshape(
        #     (1, 1, 3))).astype(np.uint8)
        # im = cv2.addWeighted(img, 0.5, label, 0.5, gamma=1)
        # cv2.imshow("vis", im)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        img = img.transpose((2, 0, 1))[::-1, :, :]
        label = label.transpose((2, 0, 1))
        img, label = (np.ascontiguousarray(x) for x in (img, label))
        return torch.tensor(img, dtype=torch.float), torch.tensor(label, dtype=torch.float)

