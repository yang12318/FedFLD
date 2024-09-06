#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python ../train.py --seed 1024 --model_arch 'resnet18' --method 'FedFLD' --dataset 'CIFAR10' --print_freq 5 --save_period 1 --n_client 10 --rule 'Dirichlet' --alpha 0.01 --active_frac 1 --bs 264 --lr 0.1 --ewc_lambda 0.7 --his_lambda 0.5