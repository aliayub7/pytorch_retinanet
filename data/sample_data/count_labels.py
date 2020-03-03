#!/usr/bin/env python3

"""Counts the number of bounding boxes for each object label"""

import argparse
from collections import defaultdict
import os
import random
import xml.etree.ElementTree as ET

def count_labels(base_dir):
    img_dir = os.path.join(base_dir, 'images')
    ann_dir = os.path.join(base_dir, 'annotations/xmls')

    for d in (img_dir, ann_dir):
        if not os.path.exists(d):
            print('Could not find {}'.format(d))
            return

    interm_map = {
        'grape_purple': 'grape',
        'grape_green': 'grape',
        'red_grapes': 'grape',
        'grapes': 'grape',
        'cherry_tomatoes': 'cherry_tomato',
        'cantalope': 'cantaloupe',
        'carrots': 'carrot',
        'celeries': 'celery',
        'apples': 'apple'
    }

    label_cnt = defaultdict(int)

    xml_filenames = sorted(os.listdir(ann_dir))
    for xidx, xml_filename in enumerate(xml_filenames):
        if not xml_filename.endswith('.xml'):
            continue
        xml_file_path = os.path.join(ann_dir, xml_filename)

        image_types = ('png', 'jpg', 'jpeg')
        name = os.path.splitext(xml_filename)[0]
        img_found = False
        for ext in image_types:
            img_path = os.path.join(img_dir, '{}.{}'.format(name, ext))
            if os.path.exists(img_path):
                img_found = True
                break

        if not img_found:
            continue

        # Count number of bounding boxes for each label
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        for node in root:
            if node.tag == 'object':
                label = node.find('name').text
                if label is None:
                    label = 'UNASSIGNED'
                elif label in interm_map:
                    label = interm_map[label]
                if node.find('bndbox') is not None:
                    label_cnt[label] += 1

    # Print number of labels per box in descending order by box count
    print('---- Boxes per label ----')
    count_getter = lambda t: t[1]
    counts = list(label_cnt.items())
    if len(counts) == 0:
        print('NO BOXES FOUND')
        return
    counts.sort(key=count_getter, reverse=True)
    maxlen = max([len(label) for label, _ in counts])
    for label, c in counts:
        print('  {} : {}'.format(label.ljust(maxlen), c))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', nargs='?', default=os.getcwd())
    args = parser.parse_args()
    count_labels(args.dir)
