#!/usr/bin/env python

import os
import random
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

    label_map_path = './{}_label_map.pbtxt'.format(prefix)
    label_dict = load_label_map(label_map_path)

    ann_dir = './annotations/xmls'
    listdataset_train_path = './{}_ann_train.txt'.format(prefix)
    listdataset_test_path = './{}_ann_test.txt'.format(prefix)

    listdataset = list()

    xml_filenames = sorted(os.listdir(ann_dir))
    for xidx, xml_filename in enumerate(xml_filenames):
        if not xml_filename.endswith('.xml'):
            continue
        xml_file_path = os.path.join(ann_dir, xml_filename)
        print('[{}/{}] {}'.format(
            xidx + 1, len(xml_filenames), xml_file_path))

        this_ann_line = xml_filename[:-4] + '.jpg'

        num_boxes = 0
        bboxes = dict()
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        for node in root:
            if node.tag == 'object':
                obj_name = node.find('name').text
                if obj_name.startswith('can'):
                    obj_name = 'can'
                xmin = int(node.find('bndbox').find('xmin').text)
                ymin = int(node.find('bndbox').find('ymin').text)
                xmax = int(node.find('bndbox').find('xmax').text)
                ymax = int(node.find('bndbox').find('ymax').text)
                if obj_name not in bboxes:
                    bboxes[obj_name] = list()
                bboxes[obj_name].append([xmin, ymin, xmax, ymax])

        for obj_name in sorted(bboxes):
            bbox_list = bboxes[obj_name]
            print(obj_name, bbox_list)
            for bidx, bbox in enumerate(bbox_list):
                xmin, ymin, xmax, ymax = bbox

                this_ann_line += ' {} {} {} {} {} {}'.format(
                    xmin, ymin, xmax, ymax, label_dict[obj_name], bidx)
                num_boxes += 1

        if num_boxes > 0:
            listdataset.append(this_ann_line)

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

    print('-- generate_listdata finished ------------------------\n')


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2:
        generate_listdata(sys.argv[1])
    else:
        print('usage:\n\t./generate_listdata.py dataset_prefix')
