r"""
Compares speed of quanization approach
"""
import argparse
import time
import os
import torch
from torch.utils.data import DataLoader
from vision.ssd.ssd import MatchPrior
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.datasets.voc_dataset import VOCDataset
from vision.ssd.data_preprocessing import TestTransform
from vision.ssd.config import mobilenetv1_ssd_config
from vision.nn.multibox_loss import MultiboxLoss

def test(loader, net, criterion, device, max_iter=None):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
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
            regression_loss, classification_loss = criterion(
                confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss
        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        if max_iter is not None:
            if iteration > max_iter:
                break
    print("Images %i, Total time: %.3f, FPS  %.3f"\
            %(n_images, total, n_images / total))
    return running_loss / num, running_regression_loss / \
        num, running_classification_loss / num


def parse_args():
    parser = argparse.ArgumentParser(
        description='Single Shot MultiBox Detector Training With Pytorch')
    parser.add_argument(
        '--resume',
        default=None,
        type=str,
        help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--validation_dataset', help='Dataset directory path')
    parser.add_argument('--batch_size', default=2, type=int,
                        help='Batch size for training')
    parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
    parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                    help='Width Multiplifier for MobilenetV2')
    args = parser.parse_args()
    return args


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


def compare_quantization(dataset, net, criterion):
    """
    Quantized vs real model
    """
    net.eval()
    device = torch.device("cpu")
    cpu_losses = test(dataset, net, criterion, device, 100)
    net.fuse_model()
    net.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare(net, inplace=True)
    test(dataset, net, criterion, device, 100)
    torch.quantization.convert(net, inplace=True)
    quant_loss = test(dataset, net, criterion, device, 100)
    return cpu_losses, quant_loss

def print_model_size(name, net):
    torch.save(net.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")/1e6
    print("Size of model %s (MB): %.3f" %(name, size))
    os.remove('temp.p')


if __name__ == "__main__":
    ARGS = parse_args()
    CONFIG = mobilenetv1_ssd_config
    DEVICE = torch.device("cpu")
    DATASET, NUM_CLASSES = prepare_data(CONFIG, ARGS)
    NET = load_model(ARGS, NUM_CLASSES, DEVICE)
    print_model_size("Full Model", NET)
    CRITERION = MultiboxLoss(
        CONFIG.priors,
        iou_threshold=0.5,
        neg_pos_ratio=3,
        center_variance=0.1,
        size_variance=0.2,
        device=DEVICE)
    cpu_loss, quant_loss = compare_quantization(DATASET, NET, CRITERION)
    print_model_size("Full Model", NET)
