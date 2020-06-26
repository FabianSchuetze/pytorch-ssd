r"""
Class to the python faces
"""

from typing import List, Tuple
import xml.etree.ElementTree as ET
import torch
import torch.utils.data as data
import cv2
import numpy as np
from .conversion_functions import Cropping, visualize_box
import matplotlib.pyplot as plt


class FacesDB(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Parameters
    ----------

    database: str
        The location of the database on disk

    target_transfrom:
        The MatchPrior transform
    """

    def __init__(self, database: str, target_transform=None,
                 random_transform=False):
        self._database = database
        self.ids = self._load_images()
        self.class_names = ('BACKGROUND', 'glabella', 'left_eye', 'right_eye',
                            'nose_tip')
        self._conversion = {'glabella': 1, 'left_eye':2, 'right_eye':3,
                            'nose_tip': 4}
        self._target_transform = target_transform
        self._Crop = Cropping()
        self._random_transform = random_transform
        self.name = 'Faces'
        self._filepath_storage = self._make_key_location_pair()

    def _load_images(self):
        tree = ET.parse(self._database)
        return tree.findall('images/image')

    def _make_key_location_pair(self):
        """
        Specifies the filename as index and the index in self.ids as value
        """
        storage = {}
        for idx, val in enumerate(self.ids):
            filename = val.get('file').rsplit('/')[-1]
            storage[filename] = idx
        return storage

    def _convert_to_box(self, box: ET.Element) -> List[int]:
        """
        Generates the bouding boxes
        """
        xmin = int(box.get('left'))
        ymin = int(box.get('top'))
        xmax = int(box.get('left')) + int(box.get('width'))
        ymax = int(box.get('top')) + int(box.get('height'))
        return [xmin, ymin, xmax, ymax]

    def _append_label(self, box: ET.Element) -> int:
        """
        Gets the corresponding label to the box
        """
        label = box.find('label').text
        return self._conversion[label]

    def _load_image(self, idx: int):
        sample = self.ids[idx]
        img = cv2.imread(sample.get('file'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _resize_img(self, img):
        img = cv2.resize(img, (300, 300))
        img = (img / 255).astype(np.float32)
        return img

    def _load_targets(self, idx, height, width):
        sample = self.ids[idx]
        boxes, labels = [], []
        for tag in sample.findall('box'):
            box = []
            for i in range(4):
                scale = width if i %2 == 0 else height
                box.append(self._convert_to_box(tag)[i] / scale)
            boxes.append(box)
            labels.append(self._append_label(tag))
        return np.array(boxes, dtype=np.float32), \
                np.array(labels, dtype=np.int64)

    def _load_sample(self, idx) -> Tuple[List]:
        img = self._load_image(idx)
        height, width, _ = img.shape
        boxes, labels = self._load_targets(idx, height, width)
        return img, boxes, labels

    def _random_augmentation(self, img, boxes, labels):
        while True:
            try:
                img, boxes, labels = self._Crop.resize(img, boxes, labels)
                break
            except IndexError:
                pass
        return img, boxes, labels

    def _transform_pipeline(self, img, boxes, labels):
        """
        Transform the images and labels randomly in a sequence of steps
        """
        rand = np.random.rand()
        if rand < 0.3:
            img, boxes, labels = self._random_augmentation(img, boxes, labels)
            img = img.astype(np.float32)
            boxes = boxes.astype(np.float32)
            labels = labels.astype(np.int64)
        # elif rand < 0.6:
            # img = img[:, ::-1].copy()
        return img, boxes, labels

    def __getitem__(self, index):
        img, boxes, labels = self._load_sample(index)
        img = self._resize_img(img)
        if self._random_transform:
            img, boxes, labels = self._transform_pipeline(img, boxes, labels)
        if self._target_transform:
            boxes, labels = self._target_transform(boxes, labels)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        return img, boxes, labels

    def __len__(self):
        return len(self.ids)

    def pull_image(self, filename: str):
        """
        Returns the original image as numpy array

        Paramters
        --------
        index: int
            The location of the image in the database

        Returns
        -------
        img: np.array
            The image
        """
        index = self._filepath_storage[filename]
        sample = self.ids[index]
        img = cv2.imread(sample.get('file'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (300, 300))
        img = (img / 255).astype(np.float32)
        return img

    def get_annotation(self, idx: int, height: int, width: int):
        """
        Returns the annotation of the image
        """
        sample = self.ids[idx]
        boxes, labels = [], []
        for tag in sample.findall('box'):
            box = []
            for i in range(4):
                scale = width if i %2 == 0 else height
                box.append(self._convert_to_box(tag)[i] * 300 / scale )
            boxes.append(box)
            labels.append(self._append_label(tag))
        difficult = np.zeros_like(labels, dtype=np.uint8)
        return idx, (np.array(boxes, dtype=np.float32), \
                    np.array(labels, dtype=np.int64), difficult)

    def get_image(self, idx: int):
        """
        Converts the image so that it can be used for prediction
        """
        img = self._load_image(idx)
        img = self._resize_img(img)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        return img

    def pull_anno(self, filename: str):
        """
        Returns the annotation of the image. In contrast to the other images,
        this function takes a string as an argument which corresponsed to the
        filename
        """
        img_id = self._filepath_storage[filename]
        img = self._load_image(img_id)
        height, width, _ = img.shape
        return self.get_annotation(img_id, height, width)[1]
