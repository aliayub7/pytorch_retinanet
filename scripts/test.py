#!/usr/bin/env python3

import sys
import os
sys.path.append("/home/guohaz/retinanet/pytorch_retinanet/src")
import argparse
import torch
import torchvision.transforms as transforms

from PIL import Image, ImageDraw, ImageFont

from pytorch_retinanet.model.retinanet import RetinaNet
from pytorch_retinanet.utils.encoder import DataEncoder
from pytorch_retinanet.config import config
from pytorch_retinanet.utils.utils import load_label_map

parser = argparse.ArgumentParser(description='Test Mashed Potato')
parser.add_argument('--img_path', default='../data/mpotato_data/images/potato-white-trail-2_0002.png',
                    help='test image path')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id


def run_test():
    print('Loading model..')
    net = RetinaNet()

    ckpt = torch.load(config.checkpoint_filename)
    net.load_state_dict(ckpt['net'])
    net.eval()
    net.cuda()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # why we need transform? which should be the correct number?
    ])

    print('Loading image..')
    img = Image.open(args.img_path)
    w, h = img.size

    print('Predicting..')
    x = transform(img)
    x = x.unsqueeze(0)
    with torch.no_grad():
        loc_preds, cls_preds = net(x.cuda())

        print('Decoding..')
        encoder = DataEncoder()
        boxes, labels, scores = encoder.decode(
            loc_preds.cpu().data.squeeze(),
            cls_preds.cpu().data.squeeze(),
            (w, h))

        label_map = load_label_map(config.label_map_filename)

        draw = ImageDraw.Draw(img, 'RGBA')
        fnt = ImageFont.truetype('Pillow/Tests/fonts/DejaVuSans.ttf', 11)
        for idx in range(len(boxes)):
            box = boxes[idx]
            label = labels[idx]
            draw.rectangle(list(box), outline=(255, 0, 0, 200))

            item_tag = '{0}: {1:.2f}'.format(
                label_map[label.item()],
                scores[idx])
            iw, ih = fnt.getsize(item_tag)
            ix, iy = list(box[:2])
            draw.rectangle((ix, iy, ix + iw, iy + ih), fill=(255, 0, 0, 100))
            draw.text(
                list(box[:2]),
                item_tag,
                font=fnt, fill=(255, 255, 255, 255))

        img.save(
            os.path.join('./rst',
                         'rst.png'),
            'PNG')


if __name__ == '__main__':
    run_test()
