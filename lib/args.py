import torchvision.datasets as dset
import torch as T
import torch
import torch.nn as nn
import os
import torch.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import sys
def get_hyperparams(args):
    batch_size = 0
    weight_decay = 0
    epochs=0
    stop_epoch = 0
    step_epoch = 0
    lr=0
    stop_lr = 0
    warm_up = 0
    num_of_outputs = 0
    lr_decay = 0
    if args.net_type in ["resnet50","resnet101"]:
        batch_size = 128
        weight_decay = 1e-4
        epochs=145
        stop_epoch = 100
        step_epoch = [30,60,90,130]
        lr=0.1
        stop_lr = 0.001
        num_of_outputs = 4
        warm_up = True
        lr_decay = 0.1
    elif args.net_type in ['resnet164','resnet110' ,'resnet1001','se_resnet164','se_resnet272','resnext29x4x16']:
        batch_size = 128
        weight_decay = 5e-4
        epochs=240
        stop_epoch = 140
        step_epoch = [60,120,200]
        lr=0.1
        stop_lr = 0.01
        num_of_outputs = 3
        warm_up = True
        lr_decay = 0.1
    elif args.net_type in ['resnet20','resnet32']:
        batch_size = 128
        weight_decay = 5e-4
        epochs=240
        stop_epoch = 140
        step_epoch = [60,120,200]
        lr=0.1
        stop_lr = 0.01
        num_of_outputs =3
        warm_up = False
        lr_decay = 0.1
    elif args.net_type in [ 'vgg16_bn',"resnet18"]:
        batch_size = 128
        weight_decay = 5e-4
        epochs=200
        stop_epoch = 140
        step_epoch = [60,120,180]
        lr=0.1
        stop_lr = 0.01
        num_of_outputs =3
        warm_up = False
        lr_decay = 0.1
    elif args.net_type in ['densenet100bc']:
        batch_size = 64
        weight_decay = 1e-4
        epochs=240
        stop_epoch = 180
        step_epoch = [80,160,220]
        lr=0.1
        stop_lr = 0.1
        num_of_outputs =3
        warm_up = False
        lr_decay = 0.1
    else:
        print("no matching net " + args.net_type)
        sys.exit()
    args.batch_size = batch_size
    args.weight_decay = weight_decay
    args.epochs = epochs
    args.stop_epoch = stop_epoch
    args.step_epoch = step_epoch
    args.lr = lr
    args.warm_up = warm_up
    args.stop_lr = stop_lr
    args.num_of_outputs = num_of_outputs
    args.lr_decay = lr_decay
    return args
    
def get_dataloader(args):
    data_train,data_test= None,None
    if args.dataset == 'cifar10':
        args.num_classes = 10
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  #鍏堝洓鍛ㄥ～鍏?锛屽湪鍚у浘鍍忛殢鏈鸿鍓垚32*32
        transforms.RandomHorizontalFlip(),  #鍥惧儚涓€鍗婄殑姒傜巼缈昏浆锛屼竴鍗婄殑姒傜巼涓嶇炕杞?        
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)), #R,G,B姣忓眰鐨勫綊涓€鍖栫敤鍒扮殑鍧囧€煎拰鏂瑰樊
    ])
        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])    
        data_train = dset.CIFAR10(root=args.data_path,train=True,transform=transform_train,download=False)
        data_test = dset.CIFAR10(root=args.data_path,train=False,transform=transform_test,download=False)

    elif args.dataset == 'cifar100':
        args.num_classes = 100
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  #鍏堝洓鍛ㄥ～鍏?锛屽湪鍚у浘鍍忛殢鏈鸿鍓垚32*32
        transforms.RandomHorizontalFlip(),  #鍥惧儚涓€鍗婄殑姒傜巼缈昏浆锛屼竴鍗婄殑姒傜巼涓嶇炕杞?       
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)), #R,G,B姣忓眰鐨勫綊涓€鍖栫敤鍒扮殑鍧囧€煎拰鏂瑰樊
    ])
        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
        data_train = dset.CIFAR100(root=args.data_path,train=True,transform=transform_train,download=False)
        data_test = dset.CIFAR100(root=args.data_path,train=False,transform=transform_test,download=False)
    elif args.dataset == 'imagenet':
        args.num_classes = 1000
        transform_train = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),  #鍥惧儚涓€鍗婄殑姒傜巼缈昏浆锛屼竴鍗婄殑姒傜巼涓嶇炕杞?        
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), #R,G,B姣忓眰鐨勫綊涓€鍖栫敤鍒扮殑鍧囧€煎拰鏂瑰樊
        ])
        transform_test = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        data_train = dset.ImageNet(root=args.data_path,split='train',transform=transform_train,download=False)
        data_test = dset.ImageNet(root=args.data_path,split='val',transform=transform_test,download=False)
    else:
        print("no matching dataset " + args.dataset)
        sys.exit()
    train_loader = DataLoader(data_train,batch_size=args.batch_size,num_workers=args.workers,shuffle=True,pin_memory=True)
    test_loader = DataLoader(data_test,batch_size=300,shuffle=False,num_workers=args.workers)
    return {'train':train_loader,'val':test_loader}