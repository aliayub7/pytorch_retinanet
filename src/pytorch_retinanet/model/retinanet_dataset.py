from __future__ import print_function
from __future__ import division

import sys
import os
import random

import torch
import torch.utils.data as data

from PIL import Image, ImageEnhance

from pytorch_retinanet.utils.encoder import DataEncoder
from pytorch_retinanet.utils.transform import resize, random_flip, random_crop, center_crop
from pytorch_retinanet.utils.utils import load_label_map
from pytorch_retinanet.utils.pt_utils import one_hot_embedding
import pytorch_retinanet.config.retinanet as config


class ListDataset(data.Dataset):
    def __init__(self,
                 img_dir=config.img_dir,
                 list_filename=config.train_list_filename,
                 label_map_filename=config.label_map_filename,
                 train=True,
                 transform=None,
                 input_size=config.img_res,
                 do_augment=True):

        self.img_dir = img_dir
        self.train = train
        self.transform = transform
        self.input_size = input_size
        self.do_augment = do_augment

        self.label_map = load_label_map(label_map_filename)

        self.img_filenames = list()
        self.boxes = list()
        self.labels = list()

        self.encoder = DataEncoder()

        with open(list_filename) as f:
            lines = f.readlines()
            self.num_samples = len(lines)

        isize = 5
        for line in lines:
            splited = line.strip().split()

            this_img_filename = splited[0]
            self.img_filenames.append(this_img_filename)

            num_boxes = (len(splited) - 1) // isize
            box = list()
            label = list()
            for bidx in range(num_boxes):
                xmin = float(splited[1 + isize * bidx])
                ymin = float(splited[2 + isize * bidx])
                xmax = float(splited[3 + isize * bidx])
                ymax = float(splited[4 + isize * bidx])
                cls = int(splited[5 + isize * bidx])
                box.append([xmin, ymin, xmax, ymax])
                label.append(cls)

            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))

    def __getitem__(self, idx):
        img_filename = self.img_filenames[idx]
        with Image.open(os.path.join(self.img_dir, img_filename)) as orig_img:
            img = orig_img.copy()
        if img.mode != 'RGB':
            img = img.convert('RGB')

        boxes = self.boxes[idx].clone()
        labels = self.labels[idx]
        size = self.input_size

        # Data augmentation
        if self.train and self.do_augment:
            img, boxes = random_flip(img, boxes)
            img, boxes = random_crop(img, boxes)
            img, boxes = resize(img, boxes, (size, size))
            if random.random() > 0.5:
                img = ImageEnhance.Color(img).enhance(random.uniform(0, 1))
                img = ImageEnhance.Brightness(img).enhance(random.uniform(0.5, 2))
                img = ImageEnhance.Contrast(img).enhance(random.uniform(0.5, 1.5))
                img = ImageEnhance.Sharpness(img).enhance(random.uniform(0.5, 1.5))
        else:
            img, boxes = resize(img, boxes, (size, size))
            # img, boxes = center_crop(img, boxes, (size, size))

        img = self.transform(img)
        return img, boxes, labels

    def collate_fn(self, batch):
        imgs = [x[0] for x in batch]
        boxes = [x[1] for x in batch]
        labels = [x[2] for x in batch]

        h = w = self.input_size
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, h, w)

        loc_targets = list()
        cls_targets = list()
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            this_loc, this_cls = self.encoder.encode(
                boxes[i], labels[i], input_size=(w, h))
            loc_targets.append(this_loc)
            cls_targets.append(this_cls)

        return inputs, torch.stack(loc_targets), torch.stack(cls_targets)

    def __len__(self):
        return self.num_samples


def test():
    print('[listdataset] test')
    ds = ListDataset(
        list_filename=config.test_list_filename,
        train=False)
    import IPython; IPython.embed()


if __name__ == '__main__':
    test()
