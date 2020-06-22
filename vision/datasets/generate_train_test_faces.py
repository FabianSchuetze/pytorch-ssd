r"""
A script to generate train and test files from the entire database.

This script should be run once to generate the train and test databases but
not more often
"""
from typing import Tuple
import argparse
import copy
import lxml.etree as ET
import numpy as np

def load_tree(path: str):
    """
    Returns the path to the tree
    """
    tree = ET.parse(path)
    return tree

def parse_args():
    """
    Converts the command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', dest='path', type=str,
                        help='Path to the xml file containing the annotations')
    args = parser.parse_args()
    return args

def split(images: ET.ElementTree, train: bool, fraction: float):
    """
    Returns the train and test images
    """
    # breakpoint();
    np.random.seed(0)
    n_imgs = len(images.findall('images/image'))
    indices = np.arange(n_imgs)
    np.random.shuffle(indices)
    cutoff = int(n_imgs * fraction)
    if train:
        cutoff = int(cutoff / 8)
    idx = 0
    keep = indices[:cutoff] if train else indices[cutoff:]
    for type_tag in images.findall('images'):
        for img in type_tag.iter('image'):
            if idx not in keep:
                type_tag.remove(img)
            idx += 1
    return images

def generate_database(images: ET.ElementTree, landmark: str)\
        -> Tuple[ET.ElementTree, ET.ElementTree]:
    """
    Generates a database for all images which contain the particular landmark
    """
    images = copy.deepcopy(images)
    for type_tag in images.findall('images/image'):
        for box in type_tag.iter('box'):
            if box.find('label').text != landmark:
                type_tag.remove(box)
    # import pdb; pdb.set_trace()
    train = split(copy.deepcopy(images), train=True, fraction=0.8)
    test = split(copy.deepcopy(images), train=False, fraction=0.8)
    return train, test


def serialize_trees(train: ET.ElementTree, test: ET.ElementTree, path: str,
                    landmark: str) -> None:
    """
    Dumps the trees to the hdd
    """
    directory = path.rsplit('/', 1)[0]
    train.write(directory + '/' + landmark + '_train.xml')
    test.write(directory + '/' + landmark + '_test.xml')

if __name__ == "__main__":
    CLARGS = parse_args()
    TREE = load_tree(CLARGS.path)
    for marker in ['left_eye', 'right_eye', 'glabella', 'nose_tip']:
        TRAIN, TEST = generate_database(TREE, marker)
        serialize_trees(TRAIN, TEST, CLARGS.path, marker)
