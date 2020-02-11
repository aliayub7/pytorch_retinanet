import torch.nn as nn
from torchvision.models.detection import FasterRCNN as _FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

# TODO: Include VGG16 backbone

# "pt" stands for "pretrained"
backbones = {
    'resnet18_fpn': lambda pt: resnet_fpn_backbone('resnet18', pt),
    'resnet34_fpn': lambda pt: resnet_fpn_backbone('resnet34', pt),
    'resnet50_fpn': lambda pt: resnet_fpn_backbone('resnet50', pt),
    'resnet101_fpn': lambda pt: resnet_fpn_backbone('resnet101', pt)
}
backbone_names = list(backbones.keys())


class FasterRCNN(_FasterRCNN):
    def __init__(self, backbone_name='resnet50_fpn', pretrained=False, **kwargs):
        """
        Creates a Faster R-CNN model. Uses torchvision as a base.

        Arguments:
            pretrained (true): Whether or not to start with a pretrained model.
            backbone_name (string): resnet18_fpn, resnet34_fpn, resnet50_fpn,
                or resnet101_fpn.

        Raises:
            ValueError: When backbone_name is not valid.
        """
        if backbone_name not in backbone_names:
            raise ValueError('{} is not a valid backbone name'.format(backbone_name))
        backbone = backbones[backbone_name](pretrained)
        # TODO: change num_classes to 1
        super(FasterRCNN, self).__init__(backbone, 91, **kwargs)
