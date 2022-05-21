import cv2
import numpy as np


def letterbox(im0, dst_size=(640, 640), interp=cv2.INTER_LINEAR, fill=114):
    r = min(dst_size[0] / im0.shape[0], dst_size[1] / im0.shape[1])
    h, w = round(im0.shape[0] * r), round(im0.shape[1] * r)
    im0 = cv2.resize(im0, (w, h), interpolation=interp)
    im = np.ones((*dst_size, 3), dtype=im0.dtype) * fill
    dh, dw = round(0.5 * (dst_size[0] - h)), round(0.5 * (dst_size[1] - w))
    im[dh:dh + h, dw:dw + w, :] = im0
    return im, r
