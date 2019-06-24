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


os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id

def confusion(prediction, truth):
    confusion_vector = prediction / truth

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives

def run_test(img, net, transform):
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
    return boxes, labels, scores


def draw_result(img, img_name, boxes, labels, scores):
    label_map = load_label_map(config.label_map_filename)

    draw = ImageDraw.Draw(img, 'RGBA')
    fnt = ImageFont.truetype('Pillow/Tests/fonts/DejaVuSans.ttf', 11)
    if boxes is not None:
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
    img.save(os.path.join('./rst', img_name + '.png'), 'PNG')


def main(path):
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

    image_filenames = sorted(os.listdir(path))
    for img_name in image_filenames:
        if img_name[-4:]=='.png':
            print('Loading image: {}'.format(img_name))
            img = Image.open(os.path.join(path, img_name))
            boxes, labels, scores = run_test(img, net, transform)
            draw_result(img, img_name, boxes, labels, scores)      


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Please provide image dir')
        os._exit(1)
    path = sys.argv[1] # get test img dir as a command line argument
    if not os.path.isdir(path):
        print('Wrong dir path')
    main(path)
    
