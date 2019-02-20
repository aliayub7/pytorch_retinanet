import sys

import torch
import torch.nn as nn

from .fpn import FPN50, FPN101

import os
config_folder = os.path.join(os.path.abspath(__file__), '../../')
print(config_folder)
sys.path.append(config_folder)
from config import config
#from pytorch_retinanet.config import config # Alternatively


class RetinaNet(nn.Module):
    def __init__(self):
        super(RetinaNet, self).__init__()

        if config.model_name == 'fpn101':
            self.fpn = FPN101()
        else:
            self.fpn = FPN50()

        self.num_anchors = 9

        self.loc_head = self._make_head(self.num_anchors * 4)
        self.cls_head = self._make_head(self.num_anchors * config.num_classes)

    def forward(self, x):
        loc_preds = list()
        cls_preds = list()

        fms = self.fpn(x)
        for fm in fms:
            loc_pred = self.loc_head(fm)
            cls_pred = self.cls_head(fm)
            # [N, 9*4, H, W] -> [N, H, W, 9*4] -> [N, H*W*9, 4]
            loc_pred = loc_pred.permute(
                0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
            # [N, 9*c, H, W] -> [N, H, W, 9*c] -> [N, H*W*9, c]
            cls_pred = cls_pred.permute(
                0, 2, 3, 1).contiguous().view(x.size(0), -1, config.num_classes)
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)

        return torch.cat(loc_preds, 1), torch.cat(cls_preds, 1)

    def _make_head(self, out_planes):
        layers = list()
        for _ in range(4):
            layers.append(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(
            nn.Conv2d(256, out_planes,
                      kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
