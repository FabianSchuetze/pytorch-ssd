r"""
An evaluation file that is compatible with how to do predicions with the cpp
client
"""
import os
from typing import List, Dict, Tuple
import numpy as np
from chainercv.evaluations import eval_detection_coco
from vision.datasets.faces import FacesDB
# from utils.augmentations import SmallAugmentation
import matplotlib.pyplot as plt
import argparse



def _convert_box(pred_box: List[List[float]]):
    """Converts box to a form that can be used by cocoeval"""
    box = np.array(pred_box)
    return box[:, [1, 0, 3, 2]]

def eval_boxes(predictions, gts):
    """Returns the coco evaluation metric for box detection.

    Parameters
    ----------
    predictions: List[Dict]
        The predictions. Length of the list indicates the number of samples.
        Each element in the list are the predictions. Keys must be 'boxes',
        'scores', and 'labels'.

    gts: List[Dict]
        The gts. Length of the list indicates the number of samples.
        Each element in the list are the predictions. Keys must be 'boxes',
        'scores', and 'labels'.

    Returns
    -------
    eval: Dict:
        The results according to the coco metric. At IoU=0.5: VOC metric.
    """
    assert len(predictions) == len(gts), "Preds and gts must have sam len"
    pred_boxes, pred_labels, pred_scores = [], [], []
    gt_boxes, gt_labels = [], []
    for pred, gt in zip(predictions, gts):
        if len(pred['boxes']) > 0:
            # breakpoint()
            pred_boxes.append(_convert_box(pred['boxes']))
            pred_labels.append(np.array(pred['labels'], dtype=np.int32))
            pred_scores.append(np.array(pred['scores']))
            gt_boxes.append(_convert_box(gt['boxes']))
            gt_labels.append(gt['labels'])
    res = eval_detection_coco(pred_boxes, pred_labels, pred_scores,
                              gt_boxes, gt_labels)
    return res, gt_labels, pred_labels

def _parse_file(filepath: str):
    boxes, labels, scores = [], [], []
    for line in open(filepath).readlines():
        elements = line.split(',')
        boxes.append([float(i) for i in elements[:4]])
        scores.append(float(elements[4]))
        labels.append(float(elements[5]))
    return {'boxes': boxes, 'labels': labels, 'scores': scores}


def load_predictions_and_gts(folder: str, dataset) -> Tuple[List[Dict]]:
    """
    parses the predicitons at path and returns a list of results
    """
    predictions = []
    gts = []
    for file in os.listdir(folder):
        prediction = _parse_file(folder + file)
        new_file = file.split('.')[0] + '.png'
        try:
            gt = dataset.pull_anno(new_file)
            gts.append({'boxes':gt[0], 'labels': gt[1]})
            predictions.append(prediction)
        except KeyError:
            pass
    assert len(predictions) == len(gts), "Preds and Gts must have same lenght"
    return predictions, gts

def barchart_frequency(gt_labels: List[np.ndarray], pred_labels: List[np.ndarray]):
    """
    Given gts and preds, the function plots a barchart with the frequency with
    which the landmarks appear
    """
    breakpoint()
    val_gt, count_gt = np.unique([len(np.unique(i)) for i in gt_labels], return_counts=True)
    val_pred, count_pred = np.unique([len(np.unique(i)) for i in pred_labels], return_counts=True)
    fig, ax = plt.subplots()
    ind = np.arange(4)
    width = 0.35         # the width of the bars
    ax.bar(ind, count_gt / count_gt.sum(), width, color='green',
           label='gt')
    ax.bar(ind + width, count_pred / count_gt.sum(), width, color='red',
           label='detector')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('1', '2', '3', '4'))
    ax.set_xlabel('Numer of Features in Image', fontsize=20)
    ax.set_title('Distribution of Features in Ground-Truth and Detector',
                 fontsize=20, fontweight='bold')
    ax.legend(prop={'size': 20})
    return fig

def parse_args():
    parser = argparse.ArgumentParser(
        description="SSD Evaluation on VOC Dataset.")
    parser.add_argument("--val_dataset", type=str,
                        help="The root directory of the dataset")
    args = parser.parse_args()
    return args


def load_dataset(args):
    """
    Returns the dataset"""
    return FacesDB(args.val_dataset)

if __name__ == "__main__":
    ARGS = parse_args()
    FACES = load_dataset(ARGS)
    PREDICTIONS, GTS = load_predictions_and_gts('cpp_client/build/results/',
                                                FACES)
    RES, GT_LABELS, PRED_LABELS = eval_boxes(PREDICTIONS, GTS)
    print(RES['coco_eval'].__str__())
    # fig = barchart_frequency(GT_LABELS, PRED_LABELS)
