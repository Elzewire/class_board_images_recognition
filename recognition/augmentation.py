import os
import random

import cv2
import numpy as np


def shear_transform(img, angle, shear, translation):
    type_border = cv2.BORDER_CONSTANT
    color_border = (255, 255, 255)

    original_image = img
    rows, cols, ch = original_image.shape

    # First: Necessary space for the rotation
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    cos_part = np.abs(M[0, 0])
    sin_part = np.abs(M[0, 1])
    new_cols = int((rows * sin_part) + (cols * cos_part))
    new_rows = int((rows * cos_part) + (cols * sin_part))

    # Second: Necessary space for the shear
    new_cols += (shear * new_cols)
    new_rows += (shear * new_rows)

    # Calculate the space to add with border
    up_down = int((new_rows - rows) / 2)
    left_right = int((new_cols - cols) / 2)

    final_image = cv2.copyMakeBorder(original_image, up_down, up_down, left_right, left_right, type_border,
                                     value=color_border)
    rows, cols, ch = final_image.shape

    # Application of the affine transform.
    M_rot = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    translat_center_x = -(shear * cols) / 2
    translat_center_y = -(shear * rows) / 2

    M = M_rot + np.float64([[0, shear, translation + translat_center_x], [shear, 0, translation + translat_center_y]])
    final_image = cv2.warpAffine(final_image, M, (cols, rows), borderMode=type_border, borderValue=color_border)
    return final_image


def augmentation(img):
    # Случайное масштабирование
    stretch = (random.random() - 0.5)
    w_stretched = max(int(img.shape[1] * (1 + stretch)), 1)
    res = cv2.resize(img, (w_stretched, img.shape[0]))

    # Cлучайный сдвиг
    # shear = (random.random() * 0.5)
    # res.append(shear_transform(img, 0, .5, 0))

    return res


def perform_augmentations():
    images_list = os.listdir('data/words/ru/')
    f = open('data/words/words.txt', 'r', encoding='utf-8')
    labels = []
    for l in f.readlines():
        labels.append((l.split(' ')[0], l.split(' ')[1]))
    f.close()

    new_labels = set()

    for k, i in enumerate(images_list):
        img = cv2.imread('data/words/ru/{}'.format(i))
        new_image = augmentation(img)
        new_labels.add(labels[k])

        new_name = '{}-{}.png'.format(i.split('.')[0], 'a')
        new_labels.add((new_name, labels[k][1]))
        cv2.imwrite('data/words/ru/{}'.format(new_name), new_image)

    f = open('data/words/words.txt', 'w', encoding='utf-8')
    for l in sorted(list(new_labels)):
        f.write('{} {}'.format(l[0], l[1]))
    f.close()
