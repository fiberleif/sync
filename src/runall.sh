#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python train.py --timesteps 5 --num_channels 39 --num_classes 3 --batch_size 256 --epochs 500 --lr 1e-1 --dropout 1
