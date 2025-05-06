#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import torch
import logging
import time

from utils.model import create_model
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Nets import MLP, CNNMnist, CNNCifar, resnet18, resnet101, resnet50, AlexNet
from models.test import test_img, test_tabular
from models.Nets import MLPAdult
from models.Update import Adult_dataloader, load_tensor, Celeba_Dataset
from utils.info import get_bias, load_models

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    logging.basicConfig(
        filename=f'./save/FL/test/{args.dataset}_{args.model}_{str(args.epochs)}_{str(args.num_users)}_{args.model}_{str(args.iid)}_{str(args.frac)}_{str(args.bs)}_{str(args.lr)}_{args.task_id}.log',
        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    global_acc = [0 for j in range(args.num_run)]
    global_loss = [0 for j in range(args.num_run)]

    for i in range(args.num_run):
        start = time.time()
        # load dataset and split users
        if args.dataset == 'mnist':
            trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
            dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
            # sample users
            if args.iid:
                dict_users = mnist_iid(dataset_train, args.num_users)
            else:
                dict_users = mnist_noniid(dataset_train, args.num_users)
        elif args.dataset == 'cifar':
            trans_cifar = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
            dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
            if args.iid:
                dict_users = cifar_iid(dataset_train, args.num_users)
            else:
                exit('Error: only consider IID setting in CIFAR10')
        elif args.dataset == 'adult' and args.model == 'mlp':
            train = load_tensor('/home/notebook/data/group/privacy/research/adult/adult.train.npz')
            test = load_tensor('/home/notebook/data/group/privacy/research/adult/adult.test.npz')

            dataset_train = Adult_dataloader(train)
            dataset_test = Adult_dataloader(test)

            if args.iid:
                dict_users = mnist_iid(dataset_train, args.num_users)
            else:
                dict_users = mnist_noniid(dataset_train, args.num_users, args.num_groups)
        elif args.dataset == 'Covid':
            train_data_transforms = transforms.Compose([
                transforms.Resize
                ((224, 224)),  # 缩放
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5],
                                     [0.5, 0.5, 0.5])])

            train_data_dir = '/home/notebook/data/group/privacy/research/Covid/'

            data_train = datasets.ImageFolder(train_data_dir, transform=train_data_transforms)
            data_test = datasets.ImageFolder(train_data_dir, transform=train_data_transforms)

            test_size = 300
            logging.info(len(data_train))
            train_size = len(data_train) - test_size
            dataset_train, dataset_test = torch.utils.data.random_split(data_train, [train_size, test_size])
            # val_ds = test_ds
            dict_users = mnist_iid(dataset_train, args.num_users)
        elif args.dataset == 'celeba':
            celeba_transform = transforms.Compose([
                transforms.Resize((224, 224)),  # 调整图像尺寸为 224x224
                transforms.ToTensor(),  # 将图像转换为张量
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])  # 标准化图像张量

            img_dir = '/home/notebook/data/group/privacy/research/celeba/img_align_celeba'
            data_train = Celeba_Dataset('/home/notebook/data/group/privacy/research/celeba/celeba-gender-train.csv',
                                        img_dir, transform=celeba_transform)
            dataset_train, _ = torch.utils.data.random_split(data_train, [int(len(data_train) * 0.6),
                                                                          len(data_train) - int(len(data_train) * 0.6)])
            data_test = Celeba_Dataset('/home/notebook/data/group/privacy/research/celeba/celeba-gender-test.csv',
                                       img_dir, transform=celeba_transform)
            dataset_test, _ = torch.utils.data.random_split(data_test, [int(len(data_test) * 0.6),
                                                                        len(data_test) - int(len(data_test) * 0.6)])
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            exit('Error: unrecognized dataset')
        img_size = dataset_train[0][0].shape

        # 自行修改此处路径，来测试不同的模型。
        net_glob = load_models('./save/FL/global_models/', args, 0, 2025216153459)[-1] # 加载第0次完整训练的最后一个epoch的全局模型

        logging.info(net_glob)

        # copy weights
        # print()
        # w_glob = net_glob.state_dict()
        
        w_glob = create_model(args)
        w_glob.load_state_dict(net_glob)
        # testing
        w_glob.eval()

        # # testing
        # net_glob.eval()

        logging.info('----------------------Test Model----------------------')
        if args.dataset == 'adult':
            acc_test, loss_test = test_tabular(net_glob, dataset_test, args)
            logging.info(f'[Test ] loss: {loss_test:6.6f} | acc: {acc_test:6.6f} .')
        else:
            acc_test, loss_test = test_img(w_glob, dataset_test, args)
            logging.info(f'[Test ] loss: {loss_test:6.6f} | acc: {acc_test:6.6f} .')

        logging.info("Testing accuracy: {:.3f}".format(acc_test))
        logging.info("Testing loss: {:.3f}".format(loss_test))

        global_acc[i] = acc_test
        global_loss[i] = loss_test

    logging.info(f'accuracy: {global_acc}')
    logging.info(f'loss: {global_loss}')
    get_bias('accuracy', global_acc, args)
    get_bias('loss', global_loss, args)
