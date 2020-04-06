# PyTorch-RetinaNet
Train _RetinaNet_ with _Focal Loss_ in PyTorch.

Reference:
[1] [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)


## Installation
```
cd YOUR_CATKIN_WS/src
git clone https://github.com/personalrobotics/pytorch_retinanet.git
cd ./pytorch_retinanet
./load_checkpoint.sh
catkin build pytorch_retinanet
source $(catkin locate)/devel/setup.bash
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


### Excluding labels
For purposes of experimentation, you may want to exclude all images containing a particular label. To do this, run `generate_listdata.py` with the following parameter
```
generate_listdata.py --exclude=<LABEL>
```


## Train the model
Change the `src/pytorch_retinanet/config/config.py` according to your machine environment and the dataset location.

Build the project and source the catkin environment
```
catkin build pytorch_retinanet
source $(catkin locate)/devel/setup.bash
cd ./script
```

### Training RetinaNet
To train a RetinaNet model
```
./train_retinanet.py
```

### Training Faster R-CNN
To train a Faster R-CNN model on a full dataset
```
./train_fasterrcnn
```

To train on a dataset excluding images with an object label
```
./train_fasterrcnn --exclude=<LABEL>
```

## Testing the model

### Testing RetinaNet
To test a RetinaNet model
```
./test_retinanet.py
```

### Testing Faster R-CNN
The following script uses Faster R-CNN on random images, displaying bounding boxes and their scores over each image
```
./test_fasterrcnn
```

To calculate summary statistics over the entire dataset
```
./test_fasterrcnn --eval
```

*TO-DO: Add a test parameter `--exclude=<LABEL>` for performing an evaluation that **excludes** images with a particular label, as done in training.*

*TO-DO: Add a test parameter `--only=<LABEL>` for performing an evaluation on **only** includes images with a particular label, as a complement to exclusion testing.*
