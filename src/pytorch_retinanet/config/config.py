''' general configurations'''

import os


gpu_id = '0'

project_dir = "/home/guohaz/retinanet/pytorch_retinanet/" #os.path.split(os.getcwd())[0]
project_prefix = 'mpotato'

available_models = {
    'fpn50': {'model_name': 'fpn50', 'conv_layer': 'resnet50'},
    'fpn101': {'model_name': 'fpn101', 'conv_layer': 'resnet101'}}

model_key = 'fpn101'

model_name = available_models[model_key]['model_name']
base_conv_layer = available_models[model_key]['conv_layer']

img_res = 600
num_classes = 1

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

pretrained_dir = os.path.join(project_dir, 'pretrained')
pretrained_filename = os.path.join(
    pretrained_dir, '{}_net.pth'.format(project_prefix))

checkpoint_filename = os.path.join(
    project_dir, 'checkpoint/{}_ckpt.pth'.format(project_prefix))

