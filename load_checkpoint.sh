#!/bin/bash

echo "load retinanet checkpoint from mthrbrn"
if [ ! -d "./checkpoint" ]; then
    mkdir -p ./checkpoint
fi
scp -r prl@mthrbrn.personalrobotics.cs.washington.edu:/mnt/hard_data/Checkpoints/pytorch_retinanet_foods/checkpoint/food_ckpt.pth ./checkpoint/
