''' Faster R-CNN configurations '''

import os

gpu_id = '0'

backbone_name = 'resnet50_fpn'
model_prefix = 'fasterrcnn_{}'.format(backbone_name)

project_dir = os.path.split(os.getcwd())[0]
project_prefix = 'sample'

img_res = 600

train_batch_size = 2
test_batch_size = 1

dataset_dir = os.path.join(
    project_dir, 'data/{}_data'.format(project_prefix))

label_map_filename = os.path.join(
    dataset_dir, '{}_label_map.pbtxt'.format(project_prefix))
img_dir = os.path.join(dataset_dir, 'images')

train_list_filename = os.path.join(
    dataset_dir, '{}_ann_train.txt'.format(project_prefix))
test_list_filename = os.path.join(
    dataset_dir, '{}_ann_test.txt'.format(project_prefix))

checkpoint_filename = os.path.join(
    project_dir, 'checkpoint/{}_ckpt.pth'.format(model_prefix))
best_ckpt_filename = os.path.join(
    project_dir, 'checkpoint/{}_best.pth'.format(model_prefix))
