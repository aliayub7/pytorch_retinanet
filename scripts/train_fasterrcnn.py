#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division

import os
import math
import shutil
import sys
import torch
import torchvision.transforms as T
import torch.multiprocessing
import warnings

from pytorch_retinanet.model.fasterrcnn import FasterRCNN
from pytorch_retinanet.model.fasterrcnn_dataset import ListDataset
from pytorch_retinanet.utils.coco_engine import train_one_epoch, evaluate, compute_loss
import pytorch_retinanet.config.fasterrcnn as config

# Prevents data loader from opening too many files.
# See https://github.com/pytorch/pytorch/issues/11201 for a description of this issue.
torch.multiprocessing.set_sharing_strategy('file_system')

# Use GPU device from config if none are specified
if not 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id

PRINT_FREQ = 500

best_loss = float('inf')

# TODO: output results to CSV file for analysis

def run_train():
    global best_loss

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if not torch.cuda.is_available():
        warnings.warn('CUDA not found!')
    start_epoch = 0  # start from epoch 0 or last epoch

    # Data
    print('Load ListDataset')
    transform = T.Compose([T.ToTensor()])

    trainset = ListDataset(
        img_dir=config.img_dir,
        list_filename=config.train_list_filename,
        label_map_filename=config.label_map_filename,
        train=True,
        transform=transform,
        input_size=config.img_res,
        do_augment=False)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=config.train_batch_size,
        shuffle=True, num_workers=8,
        collate_fn=trainset.collate_fn)

    testset = ListDataset(
        img_dir=config.img_dir,
        list_filename=config.test_list_filename,
        label_map_filename=config.label_map_filename,
        train=False,
        transform=transform,
        input_size=config.img_res)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=config.test_batch_size,
        shuffle=False, num_workers=8,
        collate_fn=testset.collate_fn)

    # Model
    net = FasterRCNN(backbone_name=config.backbone_name, pretrained=True)
    # TODO: freeze layers of pretrained networks

    # TODO: Load pretrained model if it exists

    # Load checkpoint
    if os.path.exists(config.checkpoint_filename):
        print('Loading checkpoint: {}'.format(config.checkpoint_filename))
        checkpoint = torch.load(config.checkpoint_filename)
        net.load_state_dict(checkpoint['net'])
        best_loss = min(best_loss, checkpoint['loss'])
        start_epoch = checkpoint['epoch']

    # Get best loss
    if os.path.exists(config.best_ckpt_filename):
        print('Loading best checkpoint: {}'.format(config.best_ckpt_filename))
        best_ckpt = torch.load(config.best_ckpt_filename)
        best_loss = min(best_loss, best_ckpt['loss'])

    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    net.to(device)

    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        train_one_epoch(net, optimizer, trainloader, device, epoch, print_freq=PRINT_FREQ)
        lr_scheduler.step()

    # Test
    def test(epoch):
        global best_loss

        print('\nTest')
        print('Summary statistics:')
        evaluate(net, testloader, device=device)

        test_loss = compute_loss(net, testloader, device)
        print('Loss over test set: {:.4f}'.format(test_loss))

        # Save checkpoint
        print('Save checkpoint: {}'.format(config.checkpoint_filename))
        state = {
            'net': net.module.state_dict(),
            'loss': test_loss,
            'epoch': epoch,
        }
        if not os.path.exists(os.path.dirname(config.checkpoint_filename)):
            os.makedirs(os.path.dirname(config.checkpoint_filename))
        torch.save(state, config.checkpoint_filename)

        if test_loss < best_loss:
            shutil.copy(config.checkpoint_filename, config.best_ckpt_filename)
            best_loss = test_loss

    for epoch in range(start_epoch, start_epoch + 1000):
        train(epoch)
        test(epoch)

if __name__ == '__main__':
    run_train()
