r"""
Evalution with coco metric for faces dataset
"""
import argparse
import time
from typing import List
import torch
import numpy as np
from chainercv.evaluations import eval_detection_coco
from vision.utils.misc import str2bool
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite,\
    create_mobilenetv2_ssd_lite_predictor
from vision.datasets.faces import FacesDB

def _convert_box(pred_box: List[List[float]]):
    """Converts box to a form that can be used by cocoeval"""
    box = np.array(pred_box)
    return box[:, [1, 0, 3, 2]]


def convert_gt_box(gt):
    """
    Convert the result to a useful from
    """
    boxes = gt[:, :4]
    boxes[:, 0] *= 300
    boxes[:, 2] *= 300
    boxes[:, 1] *= 300
    boxes[:, 3] *= 300
    boxes = np.array(boxes[:, [1, 0, 3, 2]])
    return boxes

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
    # breakpoint()
    for pred, gt in zip(predictions, gts):
        if len(pred['boxes']) > 0:
            pred_boxes.append(_convert_box(pred['boxes']))
            pred_labels.append(np.array(pred['labels'], dtype=np.int32))
            pred_scores.append(np.array(pred['scores']))
            gt_box = convert_gt_box(gt['boxes'])
            gt_boxes.append(gt_box)
            gt_labels.append(gt['labels'].astype(np.int32))
    # breakpoint()
    res = eval_detection_coco(pred_boxes, pred_labels, pred_scores,
                              gt_boxes, gt_labels)
    return res, gt_labels, pred_labels

def parse_args():
    """
    Returns the command line arguments
    """
    parser = argparse.ArgumentParser(
        description="SSD Evaluation on VOC Dataset.")
    parser.add_argument("--trained_model", type=str)
    parser.add_argument("--dataset", type=str,
                        help="The root directory of the dataset")
    parser.add_argument("--label_file", type=str, help="The label file path.")
    parser.add_argument("--use_cuda", type=str2bool, default=True)
    parser.add_argument("--use_2007_metric", type=str2bool, default=True)
    parser.add_argument("--nms_method", type=str, default="hard")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.5,
        help="The threshold of Intersection over Union.")
    parser.add_argument(
        "--eval_dir",
        default="eval_results",
        type=str,
        help="The directory to store evaluation results.")
    parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                        help='Width Multiplifier for MobilenetV2')
    parser.add_argument('--do_transform', default=False, type=str2bool,
                        help='Using addtional transform to the images')
    args = parser.parse_args()
    return args


def load_net(args, device):
    """
    Preparese the network
    """
    class_names = [name.strip() for name in open(args.label_file).readlines()]
    net = create_mobilenetv2_ssd_lite(len(class_names),
                                      width_mult=args.mb2_width_mult,
                                      is_test=True)
    net.load(args.trained_model)
    net = net.to(device)
    predictor = create_mobilenetv2_ssd_lite_predictor(
        net, nms_method=args.nms_method, device=device,
        do_transform=args.do_transform)
    return predictor

def obtain_results(args, device, dataset, predictor):
    """
    Retursn pred and gts
    """
    predictor = load_net(args, device)
    predictions, gts = [], []
    total_time = 0
    for i in range(len(dataset)):
        print("process image", i)
        image, gt_boxes, gt_labels = dataset[i]
        begin = time.time()
        boxes, labels, probs = predictor.predict(image)
        total_time += time.time() - begin
        predictions.append({'boxes': boxes, 'labels':labels,
                            'scores':probs})
        gts.append({'boxes':gt_boxes, 'labels':gt_labels})
    print("The were %i images passed, in %.2f second, FPS, %.2f"\
            %(len(dataset), total_time, len(dataset) / total_time))
    return predictions, gts


if __name__ == '__main__':
    ARGS = parse_args()
    DEVICE = torch.device("cpu")
    DATASET = FacesDB(ARGS.dataset)
    PREDICTOR = load_net(ARGS, DEVICE)
    PREDICTIONS, GTS = obtain_results(ARGS, DEVICE, DATASET, PREDICTOR)
    RES = eval_boxes(PREDICTIONS, GTS)[0]
    print(RES['coco_eval'].__str__())
