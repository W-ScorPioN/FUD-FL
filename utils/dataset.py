from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from models.Update import Adult_dataloader, load_tensor, Celeba_Dataset


def load_dataset2(args):
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
            dict_users = cifar_noniid(dataset_train, args.num_users)
    # elif args.dataset == 'cifar' and args.model == 'alexnet':
    #     trans_cifar = transforms.Compose([
    #         transforms.Resize((32, 32)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    #     ])
    #     # trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #     dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
    #     dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
    #     if args.iid:
    #         dict_users = cifar_iid(dataset_train, args.num_users)
    #     else:
    #         exit('Error: only consider IID setting in CIFAR10')
    elif args.dataset == 'adult' and args.model == 'mlp':
        train = load_tensor('/home/notebook/data/group/privacy/research/adult/adult.train.npz')
        test = load_tensor('/home/notebook/data/group/privacy/research/adult/adult.test.npz')

        dataset_train = Adult_dataloader(train)
        dataset_test = Adult_dataloader(test)

        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
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
        train_size = len(data_train) - test_size
        dataset_train, dataset_test = torch.utils.data.random_split(data_train, [train_size, test_size])
        # val_ds = test_ds

        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'celeba':
        # celeba_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # celeba_transform = transforms.Compose([transforms.CenterCrop((178, 178)), transforms.Resize((128, 128)), transforms.ToTensor()])
        celeba_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 调整图像尺寸为 224x224
            transforms.ToTensor(),  # 将图像转换为张量
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])  # 标准化图像张量

        # dataset_train = datasets.CelebA(root='/home/notebook/data/group/privacy/research', split='train', transform=celeba_transform, download=False)
        # dataset_valid = datasets.CelebA(root='/home/notebook/data/group/privacy/research', split='valid', transform=celeba_transform, download=False)
        # dataset_test = datasets.CelebA(root='/home/notebook/data/group/privacy/research', split='test', transform=celeba_transform, download=False)

        img_dir = '/home/notebook/data/group/privacy/research/celeba/img_align_celeba'
        data_train = Celeba_Dataset('/home/notebook/data/group/privacy/research/celeba/celeba-gender-train.csv',
                                    img_dir, transform=celeba_transform)
        dataset_train, _ = torch.utils.data.random_split(data_train, [int(len(data_train) * 0.6),
                                                                      len(data_train) - int(len(data_train) * 0.6)])
        # dataset_valid = Celeba_Dataset('/home/notebook/data/group/privacy/research/celeba/celeba-gender-valid.csv', img_dir, transform=celeba_transform)
        data_test = Celeba_Dataset('/home/notebook/data/group/privacy/research/celeba/celeba-gender-test.csv', img_dir,
                                   transform=celeba_transform)
        dataset_test, _ = torch.utils.data.random_split(data_test, [int(len(data_test) * 0.6),
                                                                    len(data_test) - int(len(data_test) * 0.6)])
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')
    return dataset_train, dataset_test, dict_users