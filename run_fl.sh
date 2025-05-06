#!/bin/bash

# 执行第一条指令并记录日志
python FL.py --model=resnet18 --dataset=cifar --num_run=2 --epochs=25 --num_user=10 --frac=1 --bs=64 --lr=0.01 --gpu=0 --num_channels=1 --iid > log1.txt 2>&1

# 执行第二条指令并记录日志
python FL.py --model=resnet18 --dataset=cifar --num_run=2 --epochs=25 --num_user=10 --frac=1 --bs=64 --lr=0.01 --gpu=0 --num_channels=1 > log2.txt 2>&1

# 执行第三条指令并记录日志
python FL.py --model=resnet18 --dataset=cifar --num_run=2 --epochs=25 --num_user=10 --frac=1 --bs=64 --lr=0.01 --gpu=0 --num_channels=1 --iid --poison_fraction=0.5 > log3.txt 2>&1

# 执行第四条指令并记录日志
python FL.py --model=resnet18 --dataset=cifar --num_run=2 --epochs=25 --num_user=10 --frac=1 --bs=64 --lr=0.01 --gpu=0 --num_channels=1 --poison_fraction=0.5 > log4.txt 2>&1

# 执行第五条指令并记录日志
python FL.py --model=resnet18 --dataset=cifar --num_run=2 --epochs=25 --num_user=50 --frac=1 --bs=64 --lr=0.01 --gpu=0 --num_channels=1 --iid > log5.txt 2>&1

# 执行第六条指令并记录日志
python FL.py --model=resnet18 --dataset=cifar --num_run=2 --epochs=25 --num_user=50 --frac=1 --bs=64 --lr=0.01 --gpu=0 --num_channels=1 > log6.txt 2>&1

# 执行第七条指令并记录日志
python FL.py --model=resnet18 --dataset=cifar --num_run=2 --epochs=25 --num_user=50 --frac=1 --bs=64 --lr=0.01 --gpu=0 --num_channels=1 --iid --poison_fraction=0.5 > log7.txt 2>&1

# 执行第八条指令并记录日志
python FL.py --model=resnet18 --dataset=cifar --num_run=2 --epochs=25 --num_user=50 --frac=1 --bs=64 --lr=0.01 --gpu=0 --num_channels=1 --poison_fraction=0.5 > log8.txt 2>&1
