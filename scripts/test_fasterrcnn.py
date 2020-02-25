#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division

import argparse
from glob import glob
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
from time import time
import torch
import torchvision.transforms as T
import torch.multiprocessing

from pytorch_retinanet.model.fasterrcnn import FasterRCNN
from pytorch_retinanet.model.fasterrcnn_dataset import ListDataset
from pytorch_retinanet.config import config
from pytorch_retinanet.utils.coco_engine import evaluate

# Prevents data loader from opening too many files.
# See https://github.com/pytorch/pytorch/issues/11201 for a description of this issue.
torch.multiprocessing.set_sharing_strategy('file_system')

os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id

# TODO: move to config
BEST_PATH = 'checkpoint/fasterrcnn_resnet50_fpn_best.pth'

project_prefix = 'food_spanet_all'

config.dataset_dir = '/mnt/hard_data/Data/foods/bite_selection_package/data/bounding_boxes_spanet_all'

config.label_map_filename = os.path.join(
    config.dataset_dir, '{}_label_map.pbtxt'.format(project_prefix))
config.img_dir = os.path.join(config.dataset_dir, 'images')

config.test_list_filename = os.path.join(
    config.dataset_dir, '{}_ann_test.txt'.format(project_prefix))

# For drawing boxes
LINE_WIDTH = 1
EDGE_COLOR = 'red'

def on_press(event):
    global show_more
    if event.key == 'enter':
        plt.close()
    elif event.key == 'escape' or event.key == 'q':
        show_more = False
        plt.close()

def run_test(do_eval):
    """
    Print AP and AR scores, and load a random image from the test set
    with predicted bounding boxes overlayed.

    Args:
        do_eval (bool): Whether to calculate average precision and recall
            over the entire test set.
    """
    global show_more

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if not torch.cuda.is_available():
        warnings.warn('CUDA not found!')

    # Data
    print('Load ListDataset')
    transform = T.Compose([T.ToTensor()])

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

    # Load best checkpoint
    net = FasterRCNN(backbone_name='resnet50_fpn', pretrained=True)
    if os.path.exists(BEST_PATH):
        print('Loading best checkpoint: {}'.format(BEST_PATH))
        ckpt = torch.load(BEST_PATH)
        net.load_state_dict(ckpt['net'])
    else:
        print('No best checkpoint found')
        exit(0)

    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    net.to(device)

    if (do_eval):
        print('\nDoing evaluation...')
        print('Loss: {:.6f}'.format(ckpt['loss']))
        if (do_eval):
            evaluate(net, testloader, device=device)

    # We do not load from the data loader since it resizes the image.
    print('\nShowing bounding box predictions on test set images')
    fnames = glob(os.path.join(config.img_dir, '*.png'))
    random.shuffle(fnames)
    show_more = True
    print('Press <ENTER> for the next image, <ESC> to quit')
    net.eval()
    size = config.img_res
    trans = T.Compose([T.ToTensor()])
    for fname in fnames:
        if not show_more:
            break

        print('    Showing {}...'.format(os.path.basename(fname)))
        fig, ax = plt.subplots()
        fig.canvas.mpl_connect('key_press_event', on_press)

        # Load image
        img = Image.open(fname)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Predict boxes
        resized_img = img.resize((size, size), Image.BILINEAR)
        resized_img = torch.stack([trans(resized_img)])
        begin = time()
        pred = net(resized_img)
        elapse = time() - begin
        print('        Boxes predicted in {:.3f}s'.format(elapse))
        pred_boxes = pred[0]['boxes']
        scores = pred[0]['scores']
        # TODO: Show scores and use IoU threshold

        # Resize boxes to fit original image
        w, h = img.size
        w_ratio = w / size
        h_ratio = h / size
        ratio = torch.Tensor(
            [w_ratio, h_ratio, w_ratio, h_ratio]
        ).to(device)
        boxes = pred_boxes * ratio

        # Draw image and boxes
        ax.imshow(np.asarray(img))
        for box, score in zip(boxes, scores):
            xmin, ymin, xmax, ymax = box
            bw = xmax - xmin
            bh = ymax - ymin
            rect = patches.Rectangle(
                (xmin, ymin), bw, bh,
                linewidth=LINE_WIDTH,
                edgecolor=EDGE_COLOR,
                facecolor='none'
            )
            ax.add_patch(rect)
            score_txt = '{:.3f}'.format(score)
            if ymin < 10:
                x, y = xmin + 3, ymin + 11
            else:
                x, y = xmin, ymin - 1
            ax.text(
                x, y, score_txt,
                fontsize=8, color=EDGE_COLOR
            )

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.show()
    print('Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show summary statistics and/or bounding boxes over test images.')
    parser.add_argument(
        '-e', '--eval',
        dest='do_eval', action='store_true',
        help='Evaluate average precision and recall'
    )
    args = parser.parse_args()

    run_test(args.do_eval)
