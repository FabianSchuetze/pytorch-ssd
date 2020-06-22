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
    parser.add_argument('--dataset', dest='dataset', type=str,
                        help='Path to the xml file containing the annotations')
    args = parser.parse_args()
    return args

def split(images: ET.ElementTree, indices: np.ndarray):
    """
    Returns the train and test images
    """
    idx = 0
    for type_tag in images.findall('images'):
        for img in type_tag.iter('image'):
            if idx not in indices:
                type_tag.remove(img)
            idx += 1
    return images

def generate_database(images: ET.ElementTree)\
        -> Tuple[ET.ElementTree, ET.ElementTree]:
    """
    Generates a database for all images which
    """
    np.random.seed(0)
    n_imgs = len(images.findall('images/image'))
    indices = np.arange(n_imgs)
    np.random.shuffle(indices)
    fraction = 0.8
    cutoff = int(len(indices) * fraction)
    train = split(copy.deepcopy(images), indices=indices[:cutoff])
    test = split(copy.deepcopy(images), indices=indices[cutoff:])
    return train, test


def serialize_trees(train: ET.ElementTree, test: ET.ElementTree, path: str) -> None:
    """
    Dumps the trees to the hdd
    """
    breakpoint()
    directory = path.rsplit('.', 1)[0]
    train.write(directory + '_train.xml')
    test.write(directory + '_test.xml')

if __name__ == "__main__":
    CLARGS = parse_args()
    TREE = load_tree(CLARGS.dataset)
    TRAIN, TEST = generate_database(TREE)
    serialize_trees(TRAIN, TEST, CLARGS.dataset)
