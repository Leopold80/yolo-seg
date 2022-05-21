import functools
import math
import random

import cv2
import numpy as np

from model import letterbox


class Compose:
    def __init__(self, transpose):
        self.transpose = transpose

    def transform(self, im, label):
        for tr in self.transpose:
            im, label = tr(im, label)
        return im, label

    def __call__(self, im, label):
        return self.transform(im, label)


class LetterBox:
    """灰条填充 随机缩放"""
    def __init__(self, size=(640, 640)):
        self.size = size

    def __call__(self, im, label):
        im, _ = letterbox(im, dst_size=self.size, interp=cv2.INTER_LINEAR, fill=114)
        label, _ = letterbox(label, dst_size=self.size, interp=cv2.INTER_LINEAR, fill=0)
        return im, label


class RandomPerspective:
    """随机仿射变换"""

    def __init__(self):
        ...

    def __call__(self, im, label):
        random_para = {
            "p1": random.uniform(-0.0001, 0.0001),
            "p2": random.uniform(-0.0001, 0.0001),
            "a": random.uniform(-3, 3),
            "scale": random.uniform(1 - 0.1, 1 + 0.1),
            "s1": random.uniform(-2, 2),
            "s2": random.uniform(-2, 2),
            "t0": random.uniform(-10, 10),
            "t1": random.uniform(-10, 10)
        }
        im = self._transform(im, random_para, fill=(114, 114, 114))
        label = self._transform(label, random_para, fill=(0, 0, 0))
        return im, label

    @staticmethod
    def _transform(im, trans, fill=(114, 114, 114)):
        """yolov5 random_perspective()"""
        para = "p1", "p2", "a", "scale", "s1", "s2", "t0", "t1"
        p1, p2, a, scale, s1, s2, t0, t1 = (trans[k] for k in para)

        M = []

        P = np.eye(3)
        P[2:3, 0:2] = np.array([p1, p2]).reshape(1, 2)
        M.append(P)  # 仿射

        R = np.eye(3)
        R[:2, :] = cv2.getRotationMatrix2D(angle=a, center=[x // 2 for x in im.shape[:2]], scale=scale)
        M.append(R)  # 旋转缩放

        S = np.eye(3)
        S[0, 1] = math.tan(s2 * math.pi / 180.)
        S[1, 0] = math.tan(s1 * math.pi / 180.)
        M.append(S)  # 剪切

        T = np.eye(3)
        T[0:2, 2:3] = np.array([t0, t1]).reshape(2, 1)
        M.append(T)  # 平移

        M = functools.reduce(lambda x, y: x @ y, reversed(M)) if len(M) > 1 else M[0]
        im = cv2.warpPerspective(im, M, im.shape[:2][::-1], borderValue=fill)
        return im



