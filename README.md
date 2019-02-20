# PyTorch-RetinaNet
Train _RetinaNet_ with _Focal Loss_ in PyTorch.

Reference:  
[1] [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)  


## Preparing a dataset
Make a symlink or put your dataset for training in `data` directory. RetinaNet currently supports annotations in xml format.

For example, if your dataset name is `sample_data`, all the images should be in `data/sample_data/images`, and annotations should be saved in `data/sample_data/annotations/xmls`. Then, define a label map (id - name pairs) in `data/sample_data/sample_label_map.pbtxt`.

Please check the `data/sample_dir` for actual exmaples. When the images, xmls, and label_map are ready, run `generate_listdata.py` in the sample directory to generate `sample_ann_train.txt` and `sample_ann_test.txt`. These two files will be used `ListDataset` (from `data/listdataset.py`).


## Train the model
Change the `config/confg.py` accordingly.

Run `run_train.sh`.
