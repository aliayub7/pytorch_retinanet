#!/usr/bin/env python

import argparse
import os
import random
import sys
import collections
import xml.etree.ElementTree as ET

LABEL_MAP = {
    'apple': 'apple',
    'apricot': None,
    'banana': 'banana',
    'bell_pepper': 'bell_pepper',
    'blackberry': None,
    'broccoli': 'broccoli',
    'cantaloupe': 'cantaloupe',
    'carrot': 'carrot',
    'celery': 'celery',
    'cherry_tomato': 'cherry_tomato',
    'egg': None,
    'grape_purple': 'grape',
    'grape_green': 'grape',
    'melon': 'melon',
    'strawberry': 'strawberry',
    'red_grapes': 'grape',
    'grapes': 'grape',
    'cherry_tomatoes': 'cherry_tomato',
    'cauliflower': 'cauliflower',
    'honeydew': 'honeydew',
    'kiwi': 'kiwi',
    'cantalope': 'cantaloupe',
    'carrots': 'carrot',
    'celeries': 'celery',
    'apples': 'apple',
}

def load_label_map(label_map_path):
    with open(label_map_path, 'r') as f:
        content = f.read().splitlines()
        f.close()
    assert content is not None, 'cannot find label map'

    temp = list()
    for line in content:
        line = line.strip()
        if (len(line) > 2 and
                (line.startswith('id') or
                 line.startswith('name'))):
            temp.append(line.split(':')[1].strip())

    label_dict = dict()
    for idx in range(0, len(temp), 2):
        item_id = int(temp[idx])
        item_name = temp[idx + 1][1:-1]
        label_dict[item_name] = item_id
    return label_dict

def write_datasets(dataset, train_path, test_path):
    num_trainset = int(len(dataset) * 0.9)
    with open(train_path, 'w') as f:
        for idx in range(0, num_trainset):
            f.write('{}\n'.format(dataset[idx]))
        f.close()
    with open(test_path, 'w') as f:
        for idx in range(num_trainset, len(dataset)):
            f.write('{}\n'.format(dataset[idx]))
        f.close()

    print('\nTraining set written to {} ({} images)'.format(train_path, num_trainset))
    num_testset = len(dataset) - num_trainset
    print('Testing set written to {} ({} images)'.format(test_path, num_testset))


def generate_listdata(prefix, exclude=None):
    label_map_path = '{}_label_map.pbtxt'.format(prefix)
    label_dict = load_label_map(label_map_path)

    print("Label Dict:")
    print(label_dict)

    img_dir = 'images'
    ann_dir = 'annotations/xmls'
    if exclude is None:
        dataset_train_path = '{}_ann_train.txt'.format(prefix)
        dataset_test_path = '{}_ann_test.txt'.format(prefix)
        dataset = list()
    else:
        exclude_prefix = '{}_no_{}'.format(prefix, exclude)
        only_prefix = '{}_only_{}'.format(prefix, exclude)
        dataset_train_path = '{}_ann_train.txt'.format(exclude_prefix)
        dataset_test_path = '{}_ann_test.txt'.format(exclude_prefix)
        dataset_only_train_path = '{}_ann_train.txt'.format(only_prefix)
        dataset_only_test_path = '{}_ann_test.txt'.format(only_prefix)
        dataset = list()
        dataset_only = list()

    xml_filenames = sorted(os.listdir(ann_dir))
    bad_boxes = 0
    total_boxes = 0
    for xidx, xml_filename in enumerate(xml_filenames):
        if not xml_filename.endswith('.xml'):
            continue
        xml_file_path = os.path.join(ann_dir, xml_filename)

        this_ann_line = xml_filename[:-4] + '.png'
        if not os.path.exists(os.path.join(img_dir, this_ann_line)):
            continue

        num_boxes = 0
        bboxes = collections.defaultdict(list)
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        do_exclude = False
        for node in root:
            if node.tag == 'object':
                # TODO: Add options for different label filters
                label = node.find('name').text.lower()
                if label in LABEL_MAP:
                    label = LABEL_MAP[label]
                if (exclude is not None) and (label == exclude):
                    do_exclude = True
                label = 'food'

                if node.find('bndbox') is None:
                    continue
                xmin = int(node.find('bndbox').find('xmin').text)
                ymin = int(node.find('bndbox').find('ymin').text)
                xmax = int(node.find('bndbox').find('xmax').text)
                ymax = int(node.find('bndbox').find('ymax').text)
                box = [xmin, ymin, xmax, ymax]

                bad_x = xmin >= xmax
                bad_y = ymin >= ymax
                if bad_x or bad_y:
                    print('Bad bounding box in "{}": {}'.format(
                        xml_filename, box))
                    if bad_x:
                        print('    xmin >= xmax')
                    if bad_y:
                        print('    ymin >= ymax')
                    bad_boxes += 1
                else:
                    bboxes[label].append(box)
                total_boxes += 1

        for label in sorted(bboxes):
            bbox_list = bboxes[label]
            if label is None:
                continue

            for bidx, bbox in enumerate(bbox_list):
                xmin, ymin, xmax, ymax = bbox
                this_ann_line += ' {} {} {} {} {}'.format(
                    xmin, ymin, xmax, ymax, label_dict[label])
                num_boxes += 1

        if num_boxes > 0:
            if do_exclude:
                dataset_only.append(this_ann_line)
            else:
                dataset.append(this_ann_line)

    good_boxes = total_boxes - bad_boxes
    print('\n{}/{} bounding boxes used'.format(total_boxes, good_boxes))

    random.shuffle(dataset)
    if exclude is not None:
        random.shuffle(dataset_only)

    write_datasets(
        dataset,
        dataset_train_path,
        dataset_test_path
    )
    if exclude is not None:
        write_datasets(
            dataset_only,
            dataset_only_train_path,
            dataset_only_test_path
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate a training and testing set')
    parser.add_argument('dataset_prefix', type=str)
    parser.add_argument(
        '--exclude', '-e', type=str, required=False,
        help='exclude all images that include boxes with this label'
    )
    args = parser.parse_args()

    if args.exclude is not None:
        print('Separate datasets will be created with and without {}'.format(args.exclude))
    print('Creating training and test sets...')
    generate_listdata(args.dataset_prefix, exclude=args.exclude)
    print('\nDone')
