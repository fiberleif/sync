#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python train.py --lr_selector 1
CUDA_VISIBLE_DEVICES=3 python train.py --lr_selector 1e-1
CUDA_VISIBLE_DEVICES=3 python train.py --lr_selector 1e-2
CUDA_VISIBLE_DEVICES=3 python train.py --lr_selector 1e-3
