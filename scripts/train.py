#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division

import os
import math
import sys
sys.path.append("/home/guohaz/retinanet/pytorch_retinanet/src")
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from pytorch_retinanet.loss.focal_loss import FocalLoss
from pytorch_retinanet.model.retinanet import RetinaNet
from pytorch_retinanet.model.retinanet_dataset import ListDataset
from pytorch_retinanet.model.init_retinanet import import_pretrained_resnet
from pytorch_retinanet.config import config


os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id

best_loss = float('inf')  # best test loss


def run_train():
    assert torch.cuda.is_available(), 'Error: CUDA not found!'
    start_epoch = 0  # start from epoch 0 or last epoch

    # Data
    print('Load ListDataset')
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    trainset = ListDataset(
        img_dir=config.img_dir,
        list_filename=config.train_list_filename,
        label_map_filename=config.label_map_filename,
        train=True,
        transform=transform,
        input_size=config.img_res)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=config.train_batch_size,
        shuffle=True, num_workers=1,
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
    net = RetinaNet()

    if os.path.exists(config.checkpoint_filename):
        print('Load saved checkpoint: {}'.format(config.checkpoint_filename))
        checkpoint = torch.load(config.checkpoint_filename)
        net.load_state_dict(checkpoint['net'])
        best_loss = checkpoint['loss']
        start_epoch = checkpoint['epoch']
    else:
        print('Load pretrained model: {}'.format(config.pretrained_filename))
        if not os.path.exists(config.pretrained_filename):
            import_pretrained_resnet()
        net.load_state_dict(torch.load(config.pretrained_filename))

    net = torch.nn.DataParallel(
        net, device_ids=range(torch.cuda.device_count()))
    net.cuda()

    criterion = FocalLoss()
    optimizer = optim.SGD(
        net.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    # optimizer = optim.Adam(net.parameters(), lr=0.001, eps=1e-3, amsgrad=True)

    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        net.module.freeze_bn()
        train_loss = 0

        total_batches = int(math.ceil(
            trainloader.dataset.num_samples / trainloader.batch_size))

        for batch_idx, targets in enumerate(trainloader):
            inputs = targets[0]
            loc_targets = targets[1]
            cls_targets = targets[2]

            inputs = inputs.cuda()
            loc_targets = loc_targets.cuda()
            cls_targets = cls_targets.cuda()

            optimizer.zero_grad()
            loc_preds, cls_preds = net(inputs)
            loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.data
            print('[%d| %d/%d] loss: %.3f | avg: %.3f' %
                  (epoch, batch_idx, total_batches,
                   loss.data, train_loss / (batch_idx + 1)))

    # Test
    def test(epoch):
        print('\nTest')
        net.eval()
        test_loss = 0

        total_batches = int(math.ceil(
            testloader.dataset.num_samples / testloader.batch_size))

        for batch_idx, targets in enumerate(testloader):
            inputs = targets[0]
            loc_targets = targets[1]
            cls_targets = targets[2]

            inputs = inputs.cuda()
            loc_targets = loc_targets.cuda()
            cls_targets = cls_targets.cuda()

            loc_preds, cls_preds = net(inputs)
            loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
            test_loss += loss.data
            print('[%d| %d/%d] loss: %.3f | avg: %.3f' %
                  (epoch, batch_idx, total_batches,
                   loss.data, test_loss / (batch_idx + 1)))

        # Save checkpoint
        global best_loss
        test_loss /= len(testloader)
        # if test_loss < best_loss:
        print('Save checkpoint: {}'.format(config.checkpoint_filename))
        state = {
            'net': net.module.state_dict(),
            'loss': test_loss,
            'epoch': epoch,
        }
        if not os.path.exists(os.path.dirname(config.checkpoint_filename)):
            os.makedirs(os.path.dirname(config.checkpoint_filename))
        torch.save(state, config.checkpoint_filename)
            # best_loss = test_loss

    for epoch in range(start_epoch, start_epoch + 150):
        train(epoch)
        test(epoch)


if __name__ == '__main__':
    run_train()
