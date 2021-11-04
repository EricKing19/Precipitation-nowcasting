import torch
import numpy as np
import numbers
import random
from scipy.ndimage.interpolation import zoom


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *img):
        for t in self.transforms:
            img = t(*img)
        return img


class ToTensor(object):
    def __call__(self, *img):
        img = list(img)
        for i in range(len(img)):
            if isinstance(img[i], np.ndarray):
                img[i] = torch.from_numpy(img[i])
            elif isinstance(img[i], list):
                for j in range(len(img[i])):
                    img[i][j] = torch.from_numpy(img[i][j])
            else:
                raise TypeError('Data should be ndarray.')
        return tuple(img)


class Crop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, *img):
        th, tw = self.size
        img = list(img)
        for i in range(len(img)):
            if isinstance(img[i], np.ndarray):
                img[i] = self.crop(img[i], 0, 0, th, tw)
            elif isinstance(img[i], list):
                for j in range(len(img[i])):
                    img[i][j] = self.crop(img[i][j], 0, 0, th, tw)

        return tuple(img)

    def crop(self, im, x_start, y_start, x_end, y_end):
        if len(im.shape) == 3:
            return im[:, x_start:x_end, y_start:y_end]
        else:
            return im[x_start:x_end, y_start:y_end]


class RandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, *img):
        th, tw = self.size
        img = list(img)
        if isinstance(img[0], np.ndarray):
            h, w = img[0].shape[-2], img[0].shape[-1]
        elif isinstance(img[0], list):
            h, w = img[0][-1].shape[-2], img[0][-1].shape[-1]
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        for i in range(len(img)):
            if isinstance(img[i], np.ndarray):
                img[i] = self.crop(img[i], y1, x1, y1 + th, x1 + tw)
            elif isinstance(img[i], list):
                for j in range(len(img[i])):
                    img[i][j] = self.crop(img[i][j], y1, x1, y1 + th, x1 + tw)

        return tuple(img)

    def crop(self, im, x_start, y_start, x_end, y_end):
        if len(im.shape) == 3:
            return im[:, x_start:x_end, y_start:y_end]
        else:
            return im[x_start:x_end, y_start:y_end]


class Normalize(object):
    def __init__(self):
        self.mean = [62.21081126207821, 10.230474850697693, 1.762178451624242, -0.48204106295086785]
        self.std = [27.46221796, 12.14330651,  7.15763795,  5.4695023]

    def __call__(self, *img):
        img = list(img)
        for i in range(len(img)):
            for j in range(len(img[i])):
                if j == 0 or j == 5:
                    continue
                img[i][j] = (img[i][j] - self.mean[j-1]) / self.std[j-1]
        return tuple(img)


class Resize(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, *img):
        img = list(img)
        for i in range(len(img)):
            for j in range(len(img[i])):
                if len(img[i][j].shape) == 3:
                    img[i][j] = zoom(img[i][j], [1, self.size[0] / img[i][j].shape[1], self.size[1] / img[i][j].shape[2]])
        return tuple(img)


class FreeScale(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, *img):
        img = list(img)
        for i in range(8):
            img.resize(self.size)


if __name__ == '__main__':
    a = np.arange(2*10*10)
    a.resize([2, 10, 10])
    a = a.astype(np.float32)
    b = a + 1
    a = [a, a, a]

    t = Compose([
        RandomCrop(5),
        ToTensor(),
    ])
    c, d = t(*[a, b])
