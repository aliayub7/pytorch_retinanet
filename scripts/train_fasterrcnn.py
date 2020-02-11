#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division

import os
import math
import sys
import torch
import torchvision.transforms as T
import warnings

from pytorch_retinanet.model.fasterrcnn import FasterRCNN
from pytorch_retinanet.model.fasterrcnn_dataset import ListDataset
from pytorch_retinanet.config import config
from pytorch_retinanet.utils.coco_engine import train_one_epoch


os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id

best_loss = float('inf')  # best test loss

# TODO: move to config
CHECKPOINT_PATH = 'checkpoint/fasterrcnn_resnet50_fpn_ckpt.pth'

# TODO: output results to CSV file for analysis

def run_train():
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
        input_size=config.img_res)
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
    net = FasterRCNN(backbone_name='resnet50_fpn', pretrained=True)
    # TODO: freeze layers of pretrained networks

    # TODO: Load pretrained model if it exists

    # Load checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        print('Loading saved checkpoint: {}'.format(CHECKPOINT_PATH))
        checkpoint = torch.load(CHECKPOINT_PATH)
        net.load_state_dict(checkpoint['net'])
        best_loss = checkpoint['loss']
        start_epoch = checkpoint['epoch']

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
        train_one_epoch(net, optimizer, trainloader, device, epoch, print_freq=100)
        lr_scheduler.step()

    # Test
    def test(epoch):
        print('\nTest')
        evaluate(net, testloader, device=device)

        # Save checkpoint
        # TODO: need metric to select best checkpoint
        global best_loss
        test_loss /= len(testloader)
        if test_loss < best_loss:
            print('Save checkpoint: {}'.format(CHECKPOINT_PATH))
            state = {
                'net': net.module.state_dict(),
                'loss': test_loss,
                'epoch': epoch,
            }
            if not os.path.exists(os.path.dirname(CHECKPOINT_PATH)):
                os.makedirs(os.path.dirname(CHECKPOINT_PATH))
            torch.save(state, CHECKPOINT_PATH)
            best_loss = test_loss

    for epoch in range(start_epoch, start_epoch + 1000):
        train(epoch)
        test(epoch)

if __name__ == '__main__':
    run_train()
