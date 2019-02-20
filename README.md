# PyTorch-RetinaNet
Train _RetinaNet_ with _Focal Loss_ in PyTorch.

Reference:  
[1] [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)  


## Preparing a dataset
Make a symlink or put your dataset for training in `data` directory. RetinaNet currently supports annotations in xml format.

For example, if your dataset name is `can_data`, then all the images should be in `data/can_data/images`, and annotations should be saved in `data/can_data/annotations/xmls`.


## Train the model
Change the `config/confg.py` accordingly.

Run `run_train.sh`.
