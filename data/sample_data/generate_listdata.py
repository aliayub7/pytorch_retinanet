#!/usr/bin/env python

import os
import random
import collections
import xml.etree.ElementTree as ET


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


def generate_listdata(prefix):
    print('-- generate_listdata ---------------------------------\n')

    label_map_path = '{}_label_map.pbtxt'.format(prefix)
    label_dict = load_label_map(label_map_path)

    print("Label Dict:")
    print(label_dict)

    img_dir = './images'
    ann_dir = './annotations/xmls'
    listdataset_train_path = './{}_ann_train.txt'.format(prefix)
    listdataset_test_path = './{}_ann_test.txt'.format(prefix)

    interm_map = {
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

    listdataset = list()

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
        for node in root:
            if node.tag == 'object':
                # TODO: Add options for different label filters
                obj_name = node.find('name').text
                obj_name = 'food'
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
                    bboxes[obj_name].append(box)
                total_boxes += 1

        for obj_name in sorted(bboxes):
            bbox_list = bboxes[obj_name]
            if obj_name is None:
                continue

            for bidx, bbox in enumerate(bbox_list):
                xmin, ymin, xmax, ymax = bbox
                this_ann_line += ' {} {} {} {} {}'.format(
                    xmin, ymin, xmax, ymax, label_dict[obj_name])
                num_boxes += 1

        if num_boxes > 0:
            listdataset.append(this_ann_line)

    good_boxes = total_boxes - bad_boxes
    print('\n{}/{} bounding boxes used'.format(total_boxes, good_boxes))

    random.shuffle(listdataset)

    num_trainset = int(len(listdataset) * 0.9)
    with open(listdataset_train_path, 'w') as f:
        for idx in range(0, num_trainset):
            f.write('{}\n'.format(listdataset[idx]))
        f.close()
    with open(listdataset_test_path, 'w') as f:
        for idx in range(num_trainset, len(listdataset)):
            f.write('{}\n'.format(listdataset[idx]))
        f.close()

    print('\nTraining set written to {} ({} images)'.format(listdataset_train_path, num_trainset))
    num_testset = len(listdataset) - num_trainset
    print('Testing set written to {} ({} images)'.format(listdataset_test_path, num_testset))

    print('\n-- generate_listdata finished ------------------------\n')


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2:
        generate_listdata(sys.argv[1])
    else:
        print('usage:\n\t./generate_listdata.py dataset_prefix')
