r"""
Functions to crop an image and the target
"""
from typing import List
from collections import namedtuple
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def _add_patch(rec, axis, color):
    width, height = rec[2] - rec[0], rec[3] - rec[1]
    patch = patches.Rectangle((rec[0], rec[1]), width, height, linewidth=1,
                              edgecolor=color, facecolor='none')
    axis.add_patch(patch)

def visualize_box(img, boxes) -> None:
    """
    Returns the list of picutres as the result
    """
    fig, axis = plt.subplots()
    axis.imshow(img)
    for rec in np.array(boxes):
        _add_patch(300*rec, axis, color='g')


def target_fits(target: List[float], size: namedtuple) -> bool:
    """
    Checks if the target is insized the size of the size
    """
    xaxis = (target[0] > size.xmin) and (target[2] < size.xmax)
    yaxis = (target[1] > size.ymin) and (target[3] < size.ymax)
    return xaxis and yaxis


Size = namedtuple('Size', 'xmin ymin xmax ymax')


class Cropping:
    """
    Crops the images in various ways
    """

    def __init__(self):
        self._rng = np.random.RandomState(0)

    def _random_size(self):
        xmin = self._rng.randint(0, 50)
        ymin = self._rng.randint(0, 50)
        width = self._rng.randint(100, 300)
        height = self._rng.randint(100, 300)
        xmax = min(300, xmin + width)
        ymax = min(300, ymin + height)
        size = Size(xmin, ymin, xmax, ymax)
        return size

    def _extract_matching_target(self, targets: List[List[float]],
                                 size: namedtuple) -> List[List[float]]:
        rescaled = np.array(targets) * 300
        fits = []
        for idx, target in enumerate(targets):
            if target_fits(rescaled[idx, ], size):
                fits.append(target)
        return fits

    def _crop_image(self, img, size):
        crop = np.ones((300, 300, 3))
        crop[size.ymin:size.ymax, size.xmin:size.xmax, :] =\
            img[size.ymin:size.ymax, size.xmin: size.xmax, :]
        return crop

    def crop(self, img: np.ndarray, target: List[List[float]]):
        """
        Crops the images and insert it into a white patch
        """
        new_sz = self._random_size()
        crop = self._crop_image(img, new_sz)
        matching_target = self._extract_matching_target(target, new_sz)
        return crop, matching_target, new_sz

    def zero_aligned_crop(self, img: np.ndarray, target: List[List[float]]):
        """Zeros aligned the boxes"""
        # breakpoint()
        cropped, new_target, size = self.crop(img, target)
        zero_aligned = np.ones((300, 300, 3))
        new_height = size.ymax - size.ymin
        new_width = size.xmax - size.xmin
        zero_aligned[:new_height, :new_width, :]\
            = cropped[size.ymin:size.ymax, size.xmin:size.xmax, :]
        new_target = np.array(new_target)
        new_target[:, 0] -= size.xmin / 300
        new_target[:, 1] -= size.ymin / 300
        new_target[:, 2] -= size.xmin / 300
        new_target[:, 3] -= size.ymin / 300
        return zero_aligned, new_target.tolist(), size

    def resize(self, img: np.ndarray, boxes: np.ndarray, labels: np.ndarray):
        """
        Takes a random crop and resized the images
        """
        target = np.hstack((boxes, labels.reshape(len(labels), 1)))
        new_img, new_target, size = self.zero_aligned_crop(img, target)
        x_scale = 300. / (size.xmax - size.xmin)
        y_scale = 300. / (size.ymax - size.ymin)
        new_target = np.array(new_target)
        new_target[:, 0] *= x_scale
        new_target[:, 1] *= y_scale
        new_target[:, 2] *= x_scale
        new_target[:, 3] *= y_scale
        cropped = new_img[:size.ymax - size.ymin, :size.xmax - size.xmin, :]
        resized_img = cv2.resize(cropped, (300, 300))
        return resized_img, target[:, :4], target[:, 4]
