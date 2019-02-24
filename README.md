# PyTorch-RetinaNet
Train _RetinaNet_ with _Focal Loss_ in PyTorch.

Reference:  
[1] [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)  


## Setup a catkin workspace and clone this repository
```
cd CATKIN_WS/src
git clone https://github.com/personalrobotics/pytorch_retinanet.git
```

## Preparing a dataset
Make a symlink or put your dataset for training in `data` directory. RetinaNet currently supports annotations in xml format.

For example, if your dataset name is `sample_data`, all the images should be in `data/sample_data/images`, and annotations should be saved in `data/sample_data/annotations/xmls`. Then, define a label map (id - name pairs) in `data/sample_data/sample_label_map.pbtxt`.

Please check the `data/sample_data` for actual exmaples. When the images, xmls, and label_map are ready, run
```
cd ./data/sample_data
generate_listdata.py sample
```
in the sample directory to generate `sample_ann_train.txt` and `sample_ann_test.txt`. These two files will be used `ListDataset` (from `data/listdataset.py`).


## Train the model
Change the `src/pytorch_retinanet/config/config.py` according to your machine environment and the dataset location.

Build and run the training script:
```
catkin build pytorch_retinanet
source $(catkin locate)/devel/setup.bash
cd ./script
./train.py
```
