''' general configurations'''

import os


gpu_id = '0'

project_dir = os.path.split(os.getcwd())[0]
project_prefix = 'can'

available_models = {
    'fpn50': {'model_name': 'fpn50', 'conv_layer': 'resnet50'},
    'fpn101': {'model_name': 'fpn101', 'conv_layer': 'resnet101'}}

model_key = 'fpn101'

model_name = available_models[model_key]['model_name']
base_conv_layer = available_models[model_key]['conv_layer']

img_res = 500
num_classes = 2

train_batch_size = 2
test_batch_size = 1

dataset_dir = os.path.join(project_dir, 'data/can_data')

label_map_filename = os.path.join(dataset_dir, 'can_label_map.pbtxt')
img_dir = os.path.join(dataset_dir, 'images')

train_list_filename = os.path.join(dataset_dir, 'can_ann_train.txt')
test_list_filename = os.path.join(dataset_dir, 'can_ann_test.txt')

pretrained_dir = os.path.join(project_dir, 'pretrained')
pretrained_filename = os.path.join(
    pretrained_dir, '{}_net.pth'.format(project_prefix))

checkpoint_filename = os.path.join(
    project_dir, 'checkpoint/{}_ckpt.pth'.format(project_prefix))

