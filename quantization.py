r"""
Compares speed of quanization approach
"""
import time
import os
import torch
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite,\
    create_mobilenetv2_ssd_lite_predictor
from vision.datasets.faces import FacesDB
from vision.ssd.config import mobilenetv1_ssd_config
from my_eval import parse_args, eval_boxes

def obtain_results(data, net, device, args, n_steps=None, prob_threshold=0.1):
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
    # if max_steps is None:
        # max_steps = len(dataset)
    # else:
        # max_steps = min(len(dataset), max_steps)
    breakpoint()
    for i in range(len(data) if n_steps is None else min(len(data), n_steps)):
        image, gt_boxes, gt_labels = data[i]
        begin = time.time()
        boxes, labels, probs = predictor.predict(image,
                                                 prob_threshold=prob_threshold,
                                                 top_k=1)
        total_time += time.time() - begin
        predictions.append({'boxes': boxes, 'labels':labels,
                            'scores':probs})
        gts.append({'boxes':gt_boxes, 'labels':gt_labels})
    print("The were %i images passed, in %.2f second, FPS, %.2f"\
            %(len(data), total_time, len(data) / total_time))
    return predictions, gts


def load_model(args, num_classes, device):
    def create_net(num): return create_mobilenetv2_ssd_lite(
        num, width_mult=args.mb2_width_mult)
    net = create_net(num_classes)
    net.load(args.resume)
    net.to(device)
    return net


def compare_quantization(train_dataset, val_dataset, net, args):
    """
    Quantized vs real model
    """
    net.eval()
    device = torch.device("cpu")
    res = obtain_results(val_dataset, net, device, args, 400, prob_threshold=0.5)
    coco = eval_boxes(res[0], res[1])[0]
    print(coco['coco_eval'].__str__())
    net.fuse_model()
    net.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(net, inplace=True)
    obtain_results(train_dataset, net, device, args, 400, prob_threshold=0.5)
    torch.quantization.convert(net, inplace=True)
    breakpoint()
    quant_res = obtain_results(val_dataset, net, device, args, 400,
            prob_threshold=0.5)
    coco_quant = eval_boxes(quant_res[0], quant_res[1])[0]
    print(coco_quant['coco_eval'].__str__())
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
    TRAIN_DATASET = FacesDB(ARGS.train_dataset)
    VAL_DATASET = FacesDB(ARGS.val_dataset)
    ARGS.do_transform = False
    NET = load_net(ARGS, DEVICE, already_quantized=False)
    print_model_size("Full Model", NET)
    RES, QUANT_RES = compare_quantization(TRAIN_DATASET, VAL_DATASET, NET, ARGS)
