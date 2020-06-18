r"""
Compares speed of quanization approach
"""
import argparse
import time
import os
import torch
from torch.utils.data import DataLoader
from vision.ssd.ssd import MatchPrior
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite,\
    create_mobilenetv2_ssd_lite_predictor
from vision.datasets.faces import FacesDB
# from vision.ssd.data_preprocessing import TestTransform
from vision.ssd.config import mobilenetv1_ssd_config
from vision.nn.multibox_loss import MultiboxLoss
from my_eval import obtain_results, parse_args

def test(loader, net, device, max_iter=None):
    net.eval()
    # # predictor = create_mobilenetv2_ssd_lite_predictor(
        # # net, nms_method=args.nms_method, device=device,
        # # do_transform=args.do_transform)
    # running_loss = 0.0
    # running_regression_loss = 0.0
    # running_classification_loss = 0.0
    num, n_images = 0, 0
    total = 0.
    for iteration, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1
        n_images += images.size(0)
        with torch.no_grad():
            begin = time.time()
            confidence, locations = net(images)
            end = time.time()
            total += end - begin
            # regression_loss, classification_loss = criterion(
                # confidence, locations, labels, boxes)
            # loss = regression_loss + classification_loss
        # running_loss += loss.item()
        # running_regression_loss += regression_loss.item()
        # running_classification_loss += classification_loss.item()
        if max_iter is not None:
            if iteration > max_iter:
                break
    print("Images %i, Total time: %.3f, FPS  %.3f"\
            %(n_images, total, n_images / total))
    # return running_loss / num, running_regression_loss / \
        # num, running_classification_loss / num


def load_model(args, num_classes, device):
    def create_net(num): return create_mobilenetv2_ssd_lite(
        num, width_mult=args.mb2_width_mult)
    net = create_net(num_classes)
    net.load(args.resume)
    net.to(device)
    return net


def prepare_data(config, args):
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)
    test_transform = TestTransform(
        config.image_size,
        config.image_mean,
        config.image_std)
    dataset = VOCDataset(args.validation_dataset,
                         transform=test_transform,
                         target_transform=target_transform,
                         is_test=True)
    num_classes = len(dataset.class_names)
    dataloader = DataLoader(dataset, args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False, drop_last=True)
    return dataloader, num_classes


def compare_quantization(dataset, net):
    """
    Quantized vs real model
    """
    breakpoint()
    net.eval()
    device = torch.device("cpu")
    test(dataset, net, device)
    # cpu_losses = test(dataset, net, criterion, device, 100)
    net.fuse_model()
    net.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare(net, inplace=True)
    net.eval()
    test(dataset, net, device)
    # obtain_results(args, device, dataset, predictor)
    torch.quantization.convert(net, inplace=True)
    test(dataset, net, device)

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
    DATASET = FacesDB(ARGS.dataset)
    DATALOADER = DataLoader(DATASET, ARGS.batch_size,
                            num_workers=ARGS.num_workers,
                            shuffle=False, drop_last=True)
    # DATASET, NUM_CLASSES = prepare_data(CONFIG, ARGS)
    # NUM_CLASSES = [name.strip() for name in open(ARGS.label_file).readlines()]
    NET = load_net(ARGS, DEVICE)
    print_model_size("Full Model", NET)
    compare_quantization(DATALOADER, NET)
    # cpu_loss, quant_loss = compare_quantization(DATASET, NET, CRITERION)
    print_model_size("Full Model", NET)
