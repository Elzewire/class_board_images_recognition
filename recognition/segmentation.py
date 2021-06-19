import cv2
import numpy as np
from sklearn.cluster import DBSCAN

from recognition.main import predict
from recognition.preprocess import thresh, apply_mask, prop_resize_cv


def segmentation(img):
    # TODO: Рассчёт значений EPS и MIN SAMPLES для каждого изображения
    # TODO: Деформация изображения для повторной кластеризации?
    mask = thresh(img)
    masked = apply_mask(img, mask)

    # Кластеризация
    X = []
    h, w = mask.shape
    for i in range(h):
        for j in range(w):
            if mask[i, j] == 0:
                X.append([i, j, masked[i, j, 0], masked[i, j, 1], masked[i, j, 2]])
    X = np.array(X)

    labels = DBSCAN(eps=17, min_samples=13).fit_predict(X)

    # Убрать шум
    Y = []
    new_labels = []
    for k, x in enumerate(X):
        if labels[k] >= 0:
            new_labels.append(labels[k])
            Y.append(x)

    new_labels = np.array(new_labels)
    Y = np.array(Y)

    # Выделение сегментов
    segments_d = {}
    for k, l in enumerate(new_labels):
        if l not in segments_d:
            segments_d[l] = []
        segments_d[l].append([Y[k, 0], Y[k, 1]])

    # Рассчёт центров
    centroids = []
    for k in segments_d:
        centroids.append(np.sum(segments_d[k], axis=0) // len(segments_d[k]))
    centroids = np.array(centroids)

    # Повторная кластеризация
    labels = DBSCAN(eps=35, min_samples=1).fit_predict(np.array([[x[0] * 3, x[1] / 2] for x in centroids]))

    # Убрать шум
    Y = []
    new_labels = []
    for k, x in enumerate(centroids):
        if labels[k] >= 0:
            new_labels.append(labels[k])
            Y.append(x)

    Y = np.array(Y)

    # Сегментация
    new_segments_d = {}
    for k, x in enumerate(segments_d):
        if labels[k] not in new_segments_d:
            new_segments_d[labels[k]] = segments_d[x]
        else:
            new_segments_d[labels[k]] = np.append(new_segments_d[labels[k]], segments_d[x], axis=0)

    segments = []
    for k in new_segments_d:
        x = np.array(new_segments_d[k])
        off_y, off_x = (min(x[:, 0]), min(x[:, 1]))
        h, w = (max(x[:, 0]) - off_y + 1, max(x[:, 1]) - off_x + 1)
        seg = np.empty((h, w), dtype='uint8')
        seg.fill(255)
        for el in x:
            seg[el[0] - off_y, el[1] - off_x] = 0
        segments.append((seg, ((off_x, off_y), (w + off_x, h + off_y)), (w, h)))

    print('Segments: {}'.format(len(segments)))

    return np.array(segments)


if __name__ == '__main__':
    img = cv2.imread('../data/boards/cut/IMG_20190218_170650.jpg')
    img = prop_resize_cv(img, (1280, 720))

    cv2.imshow('img', thresh(img))
    cv2.waitKey()
    cv2.destroyAllWindows()

    segments = segmentation(img)
    new = img.copy()
    for s in segments:
        c = [int(np.random.random() * 255) for k in range(3)]
        col = np.full((s[2][1], s[2][0], 3), c)
        col = cv2.bitwise_and(col, col, mask=(255 - s[0]))
        new[s[1][0][1]:s[1][1][1], s[1][0][0]:s[1][1][0]] = col

    cv2.imshow('img', new)
    cv2.waitKey()
    cv2.destroyAllWindows()