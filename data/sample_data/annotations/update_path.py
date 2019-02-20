#!/usr/bin/env python

import xml.etree.ElementTree as ET
import os


image_base_dir = '/mnt/hard_data/Data/soda_handoff/data/images'


def update_xml(filepath, filename, is_edit=False):
    print('>>> {}'.format(filename))
    tree = ET.parse(filepath)
    root = tree.getroot()

    for node in root:
        if node.tag == 'folder':
            print(node.tag, node.text)
            node.text = '.'
            print('-->   ', node.text)
        elif node.tag == 'filename':
            print(node.tag, node.text)
            node.text = '%s.jpg' % filename[:-4]
            print('--> ', node.text)
        elif node.tag == 'path':
            print(node.tag, node.text)
            node.text = os.path.join(image_base_dir, '%s.jpg' % filename[:-4])
            print('--> ', node.text)

    if is_edit:
        tree.write(filepath)


def update_all_xmls(base_dir, is_edit=False):
    filenames = os.listdir(base_dir)

    trainval_list = list()

    for filename in filenames:
        filepath = os.path.join(base_dir, filename)
        update_xml(filepath, filename, is_edit)
        trainval_list.append(filename[:-4])

    f_tv = open('trainval.txt.new', 'w')
    for trainval in trainval_list:
        f_tv.write('%s\n' % trainval)
    f_tv.close()


def script_main():
    print('update_path')

    import sys
    is_edit = False
    if (len(sys.argv) == 2) and (sys.argv[1] == 'true'):
        is_edit = True
    update_all_xmls('xmls', is_edit=is_edit)

    print('edit mode: ', is_edit)


if __name__ == '__main__':
    script_main()


# End of script

