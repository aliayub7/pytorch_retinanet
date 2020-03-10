#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division

import argparse
import os
import math
import shutil
from sklearn.model_selection import KFold
import sys
import torch
import torchvision.transforms as T
import torch.multiprocessing
import warnings

from pytorch_retinanet.model.fasterrcnn import FasterRCNN
from pytorch_retinanet.model.fasterrcnn_dataset import ListDataset
from pytorch_retinanet.utils.coco_engine import train_one_epoch, evaluate, compute_loss
from pytorch_retinanet.utils.utils import SubListDataset
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

def run_train(exclude=None):
    global best_loss

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if not torch.cuda.is_available():
        warnings.warn('CUDA not found!')
    start_epoch = 0  # start from epoch 0 or last epoch

    # Data
    print('Load ListDataset')
    transform = T.Compose([T.ToTensor()])

    if exclude is None:
        train_list_filename = config.train_list_filename
        ckpt_filename = config.checkpoint_filename
        best_ckpt_filename = config.best_ckpt_filename
    else:
        train_list_filename = config.get_train_list_filename_exclude(exclude)
        ckpt_filename = config.get_checkpoint_filename_exclude(exclude)
        best_ckpt_filename = config.get_best_ckpt_filename_exclude(exclude)

    if not os.path.exists(train_list_filename):
        print('Could not find dataset {}'.format(train_list_filename))
        return

    print('Using training dataset {}'.format(train_list_filename))
    trainset = ListDataset(
        img_dir=config.img_dir,
        list_filename=train_list_filename,
        label_map_filename=config.label_map_filename,
        train=True,
        transform=transform,
        input_size=config.img_res,
        do_augment=False
    )
    kfold = KFold(config.kfolds, True)
    train_splits = []
    val_splits = []
    for train_idxs, val_idxs in kfold.split(trainset):
        train_splits.append(SubListDataset(trainset, train_idxs))
        val_splits.append(SubListDataset(trainset, val_idxs))

    # Model
    net = FasterRCNN(backbone_name=config.backbone_name, pretrained=True)
    # TODO: freeze layers of pretrained networks

    # TODO: Load pretrained model if it exists

    # Load checkpoint
    if os.path.exists(ckpt_filename):
        print('Loading checkpoint: {}'.format(ckpt_filename))
        checkpoint = torch.load(ckpt_filename)
        net.load_state_dict(checkpoint['net'])
        best_loss = min(best_loss, checkpoint['loss'])
        start_epoch = checkpoint['epoch']

    # Get best loss
    if os.path.exists(best_ckpt_filename):
        print('Loading best checkpoint: {}'.format(best_ckpt_filename))
        best_ckpt = torch.load(best_ckpt_filename)
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
        trainfold = train_splits[epoch % config.kfolds]
        trainloader = torch.utils.data.DataLoader(
            trainfold,
            batch_size=config.train_batch_size,
            shuffle=True,
            num_workers=8,
            collate_fn=trainset.collate_fn
        )
        train_one_epoch(
            net, optimizer, trainloader, device, epoch,
            print_freq=PRINT_FREQ
        )
        lr_scheduler.step()

    # Validation
    def validate(epoch):
        global best_loss

        print('\nValidation')
        print('Summary statistics:')

        valfold = val_splits[epoch % config.kfolds]
        valloader = torch.utils.data.DataLoader(
            valfold,
            batch_size=config.train_batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=trainset.collate_fn
        )
        evaluate(net, valloader, device=device)
        val_loss = compute_loss(net, valloader, device)
        print('Loss over 10-fold validation set: {:.4f}'.format(val_loss))

        # Save checkpoint
        print('Save checkpoint: {}'.format(ckpt_filename))
        state = {
            'net': net.module.state_dict(),
            'loss': val_loss,
            'epoch': epoch,
        }
        if not os.path.exists(os.path.dirname(ckpt_filename)):
            os.makedirs(os.path.dirname(ckpt_filename))
        torch.save(state, ckpt_filename)

        if val_loss < best_loss:
            shutil.copy(ckpt_filename, best_ckpt_filename)
            best_loss = val_loss

    for epoch in range(start_epoch, start_epoch + 1000):
        train(epoch)
        validate(epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a Faster R-CNN model')
    parser.add_argument(
        '--exclude', '-e', type=str, required=False,
        help='exclude all images that include boxes with this label'
    )
    args = parser.parse_args()

    if args.exclude is not None:
        print('Excluding images with objects labeled {}'.format(args.exclude))
    run_train(exclude=args.exclude)
