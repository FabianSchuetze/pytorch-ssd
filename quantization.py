r"""
Compares speed of quanization approach
"""
import argparse
import time
import os
import torch
import numpy as np
from chainercv.evaluations import eval_detection_coco
from torch.utils.data import DataLoader
from vision.ssd.ssd import MatchPrior
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite,\
    create_mobilenetv2_ssd_lite_predictor
from vision.datasets.faces import FacesDB
from vision.datasets.voc_dataset import VOCDataset
from vision.ssd.config import mobilenetv1_ssd_config
from vision.nn.multibox_loss import MultiboxLoss
from my_eval import parse_args, eval_boxes

def obtain_results(dataset, net, device, args, max_steps=None,
                   prob_threshold=0.1):
    """
    Retursn pred and gts
    """
    net.eval()
    predictor = create_mobilenetv2_ssd_lite_predictor(
        net, nms_method=args.nms_method, device=device,
        do_transform=args.do_transform)
    predictor.iou_threshold = 0.3
    predictions, gts = [], []
    total_time = 0
    if max_steps is None:
        max_steps = len(dataset)
    else:
        max_steps = min(len(dataset), max_steps)
    for i in range(max_steps):
        image, gt_boxes, gt_labels = dataset[i]
        begin = time.time()
        boxes, labels, probs = predictor.predict(image,
                                                 prob_threshold=prob_threshold)
        total_time += time.time() - begin
        predictions.append({'boxes': boxes, 'labels':labels,
                            'scores':probs})
        gts.append({'boxes':gt_boxes, 'labels':gt_labels})
    print("The were %i images passed, in %.2f second, FPS, %.2f"\
            %(len(dataset), total_time, len(dataset) / total_time))
    return predictions, gts


def load_model(args, num_classes, device):
    def create_net(num): return create_mobilenetv2_ssd_lite(
        num, width_mult=args.mb2_width_mult)
    net = create_net(num_classes)
    net.load(args.resume)
    net.to(device)
    return net


def eval_boxes_voc(predictions, gts):
    pred_boxes, pred_labels, pred_scores = [], [], []
    gt_boxes, gt_labels = [], []
    # breakpoint()
    for pred, gt in zip(predictions, gts):
        if len(pred['boxes']) > 0:
            pred 
            pred_boxes.append(pred['boxes'][pred['scores'] > 0.1])
            pred_labels.append(np.array(pred['labels'][pred['scores'] > 0.1], dtype=np.int32))
            pred_scores.append(np.array(pred['scores'][pred['scores'] > 0.1]))
            gt_boxes.append(gt['boxes'])
            gt_labels.append(gt['labels'].astype(np.int32))
    res = eval_detection_coco(pred_boxes, pred_labels, pred_scores,
                              gt_boxes, gt_labels)
    return res



def compare_quantization(dataset, net, args):
    """
    Quantized vs real model
    """
    breakpoint()
    net.eval()
    device = torch.device("cpu")
    res = obtain_results(dataset, net, device, args, 400, prob_threshold=0.2)
    if args.dataset_type == 'voc':
        coco = eval_boxes_voc(res[0], res[1])
    else:
        coco = eval_boxes(res[0], res[1])[0]
    print(coco['coco_eval'].__str__())
    net.fuse_model()
    net.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(net, inplace=True)
    obtain_results(dataset, net, device, args, 400, prob_threshold=0.2)
    # obtain_results(args, device, dataset, predictor)
    torch.quantization.convert(net, inplace=True)
    quant_res = obtain_results(dataset, net, device, args, 400,
            prob_threshold=0.2)
    if args.dataset_type == 'voc':
        coco_quant = eval_boxes_voc(quant_res[0], quant_res[1])
    else:
        coco_quant = eval_boxes(quant_res[0], quant_res[1])[0]
    print(coco_quant['coco_eval'].__str__())
    breakpoint()
    return res, quant_res

def load_net(args, device, already_quantized=True):
    """
    Preparese the network
    """
    class_names = [name.strip() for name in open(args.label_file).readlines()]
    net = create_mobilenetv2_ssd_lite(len(class_names),
                                      width_mult=args.mb2_width_mult,
                                      is_test=True)
    if already_quantized:
        net.fuse_model()
        net.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare_qat(net, inplace=True)
        torch.quantization.convert(net, inplace=True)
        net.load_state_dict(torch.load(args.trained_model))
        net.to(device)
        return net
    net.load(args.trained_model)
    net = net.to(device)
    return net
    # predictor = create_mobilenetv2_ssd_lite_predictor(
        # net, nms_method=args.nms_method, device=device,
        # do_transform=args.do_transform)
    # return predictor

def print_model_size(name, net):
    torch.save(net.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")/1e6
    print("Size of model %s (MB): %.3f" %(name, size))
    os.remove('temp.p')


if __name__ == "__main__":
    # breakpoint()
    ARGS = parse_args()
    CONFIG = mobilenetv1_ssd_config
    DEVICE = torch.device("cpu")
    if ARGS.dataset_type == "voc":
        DATASET = VOCDataset(ARGS.val_dataset, is_test=True)
    elif ARGS.dataset_type == 'faces':
        DATASET = FacesDB(ARGS.val_dataset)
        ARGS.do_transform = False
    else:
        raise NameError("Not the correct name")

    DATALOADER = DataLoader(DATASET, ARGS.batch_size,
                            num_workers=ARGS.num_workers,
                            shuffle=False, drop_last=True)
    NET = load_net(ARGS, DEVICE, already_quantized=False)
    print_model_size("Full Model", NET)
    RES, QUANT_RES = compare_quantization(DATASET, NET, ARGS)
    # # cpu_loss, quant_loss = compare_quantization(DATASET, NET, CRITERION)
    # print_model_size("Full Model", NET)
