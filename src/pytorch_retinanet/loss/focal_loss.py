from __future__ import print_function
from __future__ import division

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_retinanet.utils.pt_utils import one_hot_embedding
from pytorch_retinanet.config import config


class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def focal_loss(self, x, y):
        alpha = 0.25
        gamma = 2

        print("Y data: " + str(torch.max(y.data)))
        t = one_hot_embedding(y.data, 1 + config.num_classes)  # [N,21]
        t = t[:, 1:]  # exclude background
        t = t.cuda()  # [N,20]

        p = x.sigmoid()
        pt = p * t + (1 - p) * (1 - t)         # pt = p if t > 0 else 1-p
        # w = alpha if t > 0 else 1-alpha
        w = alpha * t + (1 - alpha) * (1 - t)
        w = w * (1 - pt).pow(gamma)
        w = w.detach()  # for torch >= 1.0
        return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.data.sum().type(torch.cuda.FloatTensor)

        if num_pos.item() == 0:
            print("BAD NUM POS!")

        mask = pos.unsqueeze(2).expand_as(loc_preds)       # [N,#anchors,4]
        m_loc_preds = loc_preds[mask].view(-1, 4)      # [#pos,4]
        m_loc_targets = loc_targets[mask].view(-1, 4)  # [#pos,4]
        loc_loss = F.smooth_l1_loss(
            m_loc_preds, m_loc_targets, size_average=False)

        pos_neg = cls_targets > -1  # exclude ignored anchors
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        m_cls_preds = cls_preds[mask].view(-1, config.num_classes)
        cls_loss = self.focal_loss(m_cls_preds, cls_targets[pos_neg])

        print('loc: {0:.03f} | cls: {1:.03f}'.format(
              loc_loss.data / num_pos,
              cls_loss.data / num_pos), end=' | ')

        loss = (loc_loss + cls_loss) / num_pos
        return loss
