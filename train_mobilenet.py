r"""
Training a (very) reduced size of imaenet
"""
import time
import argparse
import os
import logging
import sys
import copy
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from vision.nn.mobilenet_v2 import MobileNetV2
# from vision.nn.tutorial_mobilenetv2 import MobileNetV2
from vision.utils.meter import AverageMeter, accuracy
from vision.datasets.cifar import get_train_dataloader, get_test_dataloader

def prepare_data_loaders(data_path, train_batch_size, eval_batch_size):

    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    dataset_test = torchvision.datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size,
        sampler=train_sampler, num_workers=4, pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=eval_batch_size,
        sampler=test_sampler, num_workers=4, pin_memory=True)

    return data_loader, data_loader_test

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch):
    model.train()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    avgloss = AverageMeter('Loss', '1.5f')
    total_loss = 0

    cnt = 0
    for image, target in data_loader:
        start_time = time.time()
        cnt += 1
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))
        avgloss.update(loss, image.size(0))
        total_loss += loss.item()
    logging.info(
        f"Train Epoch: {epoch}, " +
        f"Top1 Accuracy: {top1.avg:.1f}, " +
        f"Top5 Accuracy {top5.avg:.1f}, " +
        f"Loss {avgloss.avg:.1f}")

def evaluate(model, criterion, data_loader, device, max_steps=None):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    # breakpoint()
    with torch.no_grad():
        for image, target in data_loader:
            image, target = image.to(device), target.to(device)
            output = model(image)
            loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            if max_steps is not None and cnt > max_steps:
                break
    return top1, top5

def quantize_evaluate(model, criterion, data_loader):
    device = torch.device('cpu')
    model.to(device)
    start_time = time.time()
    top1, top5 = evaluate(model, criterion, data_loader, device)
    end_time = time.time() - start_time
    model.fuse_model()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    evaluate(model, criterion, data_loader, device)
    torch.quantization.convert(model, inplace=True)
    start_time = time.time()
    top1_quant, top5_quant = evaluate(model, criterion, data_loader, device)
    end_time_quant = time.time() - start_time
    return (top1, top5), (top1_quant, top5_quant), (end_time, end_time_quant)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Single Shot MultiBox Detector Training With Pytorch')

    parser.add_argument(
        "--dataset_type",
        default="voc",
        type=str,
        help='Specify dataset type. Currently support voc and open_images.')
    parser.add_argument('--checkpoint_folder', default='models/', type=str)
    parser.add_argument('--val_freq', default=5, type=int)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ARGS = parse_args()
    if ARGS.dataset_type == 'imagenet':
        DATA_PATH = '/home/fabian/data/imagenet_1k'
        DATA_LOADER, DATA_LOADER_TEST = prepare_data_loaders(DATA_PATH,
                                                             train_batch_size=32,
                                                             eval_batch_size=32)
    elif ARGS.dataset_type == 'cifar':
        DATA_PATH = '/home/fabian/data/cifar'
        DATA_LOADER = get_train_dataloader(DATA_PATH, batch_size=32,
                                           num_workers=4)
        DATA_LOADER_TEST = get_test_dataloader(DATA_PATH, batch_size=32,
                                               num_workers=4)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CRITERION = nn.CrossEntropyLoss()
    NET = MobileNetV2()
    NET.to(DEVICE)
    OPTIMIZER = torch.optim.SGD(NET.parameters(), lr=0.0001, momentum=0.9,
                                weight_decay=0.0005)
    for epoch in range(40):
        train_one_epoch(NET, CRITERION, OPTIMIZER, DATA_LOADER, DEVICE, epoch)
        if epoch > 0 and epoch % ARGS.val_freq == 0:
            res, quant_res, duration =\
                quantize_evaluate(copy.deepcopy(NET), CRITERION, DATA_LOADER_TEST)
            top1, top5 = res
            top1_quant, top5_quant = quant_res
            logging.info(
                f"Epoch: {epoch}, " +
                f"Top1 Accuracy: {top1.avg:.1f}, " +
                f"Top5 Accuracy: {top5.avg:.1f}, " +
                f"Top1 Quant Accuracy: {top1_quant.avg:.1f}, " +
                f"Top5 Quant Accuracy: {top5_quant.avg:.1f}, " +
                f"Time: {duration[0]:.1f}, " +
                f"Time Quant {duration[1]:.1f}")
            # model_path = os.path.join(ARGS.checkpoint_folder,
                # f"MobileNetV2-Epoch-{epoch}-Top1-{top1.avg}.pth")
            # NET.save(model_path)
