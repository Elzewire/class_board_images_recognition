import os

import cv2
import numpy as np


def prop_resize_pil(img, size):
    # Пропорциональное сжатие/растяжение
    wt, ht = size
    w, h = img.size
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    newSize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1))
    res = img.resize(newSize)
    return res


def prop_resize_cv(img, size):
    # Пропорциональное сжатие/растяжение
    wt, ht = size
    h, w, _ = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    newSize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1))
    res = cv2.resize(img, newSize)
    return res


def apply_mask(img, mask):
    # Вырезать изображение по маске
    res = img.copy()
    h, w, _ = img.shape
    for i in range(h):
        for j in range(w):
            if mask[i, j] > 0:
                res[i, j] = mask[i, j]
    return res


def avg_gray(img):
    # Среднее значение серого
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


def thresh(img, inv_thresh=100):
    # Пороговая бинаризация
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Инвертирование для зелёных и чёрных досок
    if avg_gray(img) < inv_thresh:
        gray = 255 - gray

    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)

    return mask


if __name__ == '__main__':
    img = cv2.imread('../data/boards/cut/IMG_20190429_100352.jpg')
    mask = thresh(img)
    masked = apply_mask(img, mask)

    cv2.imshow('img', masked)
    cv2.waitKey()
    cv2.destroyAllWindows()

    img = cv2.imread('../data/boards/cut/IMG_20190410_173947.jpg')
    cv2.imshow('img', thresh(img))
    cv2.waitKey()
    cv2.destroyAllWindows()
