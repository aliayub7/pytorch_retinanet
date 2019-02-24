#!/usr/bin/env python

import math
import os
import sys

import torch
import torch.nn as nn
import torchvision.models as models

from pytorch_retinanet.model.fpn import FPN50, FPN101
from pytorch_retinanet.model.retinanet import RetinaNet
from pytorch_retinanet.config import config


def import_pretrained_resnet():
    if config.base_conv_layer == 'resnet50':
        conv_layer = models.resnet50(pretrained=True)
    elif config.base_conv_layer == 'resnet101':
        conv_layer = models.resnet101(pretrained=True)

    print('Loading pretrained {} into {}.'.format(
        config.base_conv_layer,
        config.model_name))

    if config.model_name == 'fpn101':
        fpn = FPN101()
    else:
        fpn = FPN50()

    conv_layer_states = conv_layer.state_dict()
    fpn_states = fpn.state_dict()

    for k in conv_layer_states.keys():
        if not k.startswith('fc'):  # skip fc layers
            fpn_states[k] = conv_layer_states[k]

    print('Initializing RetinaNet with {}.'.format(config.model_name))
    net = RetinaNet()
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    pi = 0.01
    nn.init.constant_(net.cls_head[-1].bias, -1 * math.log((1 - pi) / pi))

    net.fpn.load_state_dict(fpn_states)

    if not os.path.exists(os.path.dirname(config.pretrained_filename)):
        os.makedirs(os.path.dirname(config.pretrained_filename))
    torch.save(net.state_dict(), config.pretrained_filename)
    print('Finished.\nSaved in: {}'.format(config.pretrained_filename))


if __name__ == '__main__':
    import_pretrained_resnet()
