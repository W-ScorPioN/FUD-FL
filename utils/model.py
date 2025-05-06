import numpy as np
import torch

from models.Nets import MLP, CNNMnist, CNNCifar, resnet18, resnet101, resnet50, AlexNet
from models.Nets import MLPAdult
from utils.dataset import load_dataset2


def create_model(args):
    dataset_train, _, _ = load_dataset2(args)
    img_size = dataset_train[0][0].shape

    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'resnet18' and args.dataset == 'cifar':
        net_glob = resnet18(10, False).to(args.device)
    elif args.model == 'mlp' and args.dataset == 'adult':
        net_glob = MLPAdult().to(args.device)
    elif args.model == 'alexnet' and args.dataset == 'cifar':
        net_glob = AlexNet(10).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    elif args.model == 'resnet101' and args.dataset == 'Covid':
        net_glob = resnet101(3, False).to(args.device)
    elif args.model == 'resnet101' and args.dataset == 'celeba':
        net_glob = resnet101(2, False).to(args.device)
    elif args.model == 'resnet50' and args.dataset == 'celeba':
        net_glob = resnet50(2, False).to(args.device)
        # 加载预训练的 ResNet-101 模型
        # net_glob = models.resnet101(pretrained=True)

        # 获取最后一层全连接层
        # fc_in_features = net_glob.fc.in_features
        # 修改全连接层的输出维度
        # num_attributes = 1  # 二分类任务，所以输出维度为 1
        # net_glob.fc = torch.nn.Linear(fc_in_features, num_attributes)
    else:
        exit('Error: unrecognized model')

    return net_glob
