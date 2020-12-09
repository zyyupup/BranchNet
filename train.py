import torchvision
import numpy as np
from torch.optim import lr_scheduler
import time
import copy
from torch.backends import cudnn
import torchvision.datasets as dset
import torch
import torch.nn as nn
import os
import sys
import argparse
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model.imagenet_resnet import resnet50,resnet101,resnet18
from model.cifar_resnet_in_paper import resnet110,resnet20,resnet164,resnet1001,resnet32
from model.densenet import densenet100bc
from model.resnext import resnext29x4x16,resnext29x8x64,resnext29x16x64
from model.cifar_vgg import vgg16_bn
from model.cifar_se_resnet import se_resnet164
from lib.utils import progress_bar,soft_target_loss,_make_st_criterion
from lib.utils import Collabrative_Loss as CL
from lib.args import get_hyperparams,get_dataloader



parser = argparse.ArgumentParser(description='BranchNet Training')
parser.add_argument('--data_path',default='/home/zyy/data/CIFAR100/', type=str, help='path to dataset')
parser.add_argument("--dataset",default='cifar100',type = str,help = 'cifar10, cifar100 or imagenet')
parser.add_argument("--num_classes",default=100,type=int,help='number of classes')
parser.add_argument('-init','--initial_hyperparams',action='store_false',default = True,help = 'if true, use the hyperparams in lib/args. (default True.) ')
parser.add_argument('--net_type', default='resnet20', type=str, help='network type: e.g.,[cifar:resnet20,resnet110,resnet164,resnet1001,imagenet:resnet50]')
parser.add_argument('-j', '--workers', default=8, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=120, type=int, help='number of total epochs to run')
parser.add_argument('--stop_epoch', default=80, type=int, help='the start training epoch of retrain process')
parser.add_argument('--step_epoch', default=[60,100], type=list, help='divide lr by 10 in step_epoch')
parser.add_argument('--stop_lr', default=0.1, type=float, help='the lr decay of retrain process')
parser.add_argument("--cur_epoch",default = 1,type=int, help="current epoch(will be changed by resume)")
parser.add_argument('-b', '--batch_size', default=128, type=int, help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float, help='initial learning rate (default: 0.1)')
parser.add_argument('--lr_decay',default = 0.1,type=float,help='lr decay rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay (default: 5e-4)')
parser.add_argument('-warm_up',action='store_true',default = False,help = 'warm_up(set lr 0.01 in first 5 epochs)')
parser.add_argument("--resume",default="",type=str,help = "resume saved models")
parser.add_argument("--num_of_outputs",default=3,type=int,help='len of outputs, e.g., resnet164 is 3 (branch unit 1, branch unit 2, and backbone)')
parser.add_argument('--gpu_idx', default='0', type=str, help='gpu idx')
parser.add_argument('--save_interval',default = 30,type=int,help='internal to save checkpoints')
parser.add_argument('--scale_factor',default = 1,type=float,help='scale factor of branch unit')
args = parser.parse_args()
if args.initial_hyperparams:
    args = get_hyperparams(args)
#load dataset
dataloaders= get_dataloader(args)
print(args)
print("finishded loding dataset.")
#load model
if args.net_type == 'resnet18':
    net = resnet18(num_classes = args.num_classes,scale_factor = args.scale_factor)
elif args.net_type == 'resnet50':
    net = resnet50(num_classes = args.num_classes,scale_factor = args.scale_factor)
elif args.net_type == 'resnet101':
    net = resnet101(num_classes = args.num_classes,scale_factor = args.scale_factor)
elif args.net_type == 'resnet20':
    net = resnet20(num_classes=args.num_classes,scale_factor = args.scale_factor)
elif args.net_type == 'resnet32':
    net = resnet32(num_classes=args.num_classes,scale_factor = args.scale_factor)
elif args.net_type == 'resnet110':
    net = resnet110(num_classes=args.num_classes,scale_factor = args.scale_factor)
elif args.net_type == 'resnet164':
    net = resnet164(num_classes=args.num_classes,scale_factor = args.scale_factor)
elif args.net_type == 'resnet1001':
    net = resnet1001(num_classes=args.num_classes,scale_factor = args.scale_factor)
elif args.net_type == 'densenet100bc':
    net = densenet100bc(num_classes=args.num_classes)
elif args.net_type == 'resnext29x4x16':
    net = resnext29x4x16(num_classes=args.num_classes) 
elif args.net_type == 'vgg16_bn':
    net = vgg16_bn(num_classes=args.num_classes,scale_factor = args.scale_factor)  
elif args.net_type == 'se_resnet164':
    net = se_resnet164(num_classes=args.num_classes,scale_factor = args.scale_factor) 
elif args.net_type == 'se_resnet272':
    net = se_resnet272(num_classes=args.num_classes,scale_factor = args.scale_factor) 
else:
    print("no matching net.")
    sys.exit()
os.environ['CUDA_VISIBLE_DEVICES']= args.gpu_idx
if len(args.gpu_idx) >1 :
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    
net = net.cuda()

print("model params:{}".format(sum(p.numel() for p in net.parameters())/(1000*1000)))

#loss function and optimizer
criterion = nn.CrossEntropyLoss()
soft_loss = soft_target_loss(num_classes = args.num_classes)
optimizer = torch.optim.SGD(net.parameters(),lr = args.lr, momentum=args.momentum,weight_decay=args.weight_decay,nesterov=False)
acc_save = []
best_acc = 0.0
best_model_weights = 0
save_path = 'checkpoint/'+args.net_type+"/"+args.dataset
#resume

if args.resume != "":
    state = torch.load(args.resume)
    checkpoint,best_acc,optimizer_state = state['net'],state['acc'],state['optimizer']
    args.cur_epoch = state['epoch'] + 1
    optimizer.load_state_dict(optimizer_state)
    net.load_state_dict(checkpoint) 
    #net.load_state_dict({k.replace('module.',''):v for k,v in checkpoint.items()})
    acc_save = state['acc_save']
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    top_1 = [0.0 for i in range(args.num_of_outputs)]
    top_5 = [0.0 for i in range(args.num_of_outputs)]
    total = 0
    for batch_idx, (inputs, targets) in enumerate(dataloaders['train']):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        out = net(inputs)
        loss_sl = []
        loss_st = 0
        loss_cl = 0
        loss_kd = 0
        predicts = []
        predicts_5 = []
        for x in out:
            loss_sl.append(criterion(x,targets))
            predicts.append(x.max(1)[1])
            predicts_5.append(x.topk(5)[1])

        for i,x in enumerate(out[:-1]):
            loss_cl+= sum([CL(out[i],out[j]) for j in range(i+1,len(out))])
        if epoch >= args.stop_epoch:
            loss_st = soft_loss(out[0],targets,nn.Softmax(dim=1)(out[-1]))
        loss = torch.sum(torch.stack(loss_sl,dim=0)) + loss_cl + loss_st
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        total+= targets.size(0)
        for i,(pre,pre_5) in enumerate(zip(predicts,predicts_5)):
            top_1[i] += pre.eq(targets).sum().item()
            top_5[i] += pre_5.t().eq(targets).sum().item()
        progress_bar(batch_idx, len(dataloaders['train']), 'Loss: %.3f | Acc_1: %.3f%% | Acc_5: %.3f%%'
            % (train_loss/((len(top_1)-1)*(batch_idx+1)), 100.*top_1[-1]/total,100.*top_5[-1]/total))
    train_info = [epoch,train_loss/((len(top_1)-1)*(batch_idx+1))]+[float(i)/total for i in top_1]+[float(i)/total for i in top_5]
    acc_save.append(train_info)
def test(epoch):
    global best_acc,acc_save,best_model_weights
    net.eval()
    test_loss = 0
    top_1 = [0.0 for i in range(args.num_of_outputs)]
    top_5 = [0.0 for i in range(args.num_of_outputs)]
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloaders['val']):
            inputs, targets = inputs.cuda(), targets.cuda()
            out = net(inputs)
            loss = []
            predicts = []
            predicts_5 = []
            for x in out:
                predicts.append(x.max(1)[1])
                predicts_5.append(x.topk(5)[1])
            loss = criterion(out[-1],targets)
            test_loss += loss.item()
            total+= targets.size(0)
            for i,(pre,pre_5) in enumerate(zip(predicts,predicts_5)):
                top_1[i] += pre.eq(targets).sum().item()
                top_5[i] += pre_5.t().eq(targets).sum().item()
            progress_bar(batch_idx, len(dataloaders['val']), 'Loss: %.3f | Acc: %.3f%% | Acc_5: %.3f%%'
                % (test_loss/((len(top_1)-1)*(batch_idx+1)), 100.*top_1[-1]/total,100.*top_5[-1]/total))
        test_info = [epoch,test_loss/((len(top_1)-1)*(batch_idx+1))]+[i/total for i in top_1]+[i/total for i in top_5]
        acc_save.append(test_info)
    # Save checkpoint.
    acc = 100.*top_1[-1]/total
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if acc > best_acc:
        best_acc = acc
        print('Saving..')
        best_model_weights = net.state_dict()
    state = {
            'net': net.state_dict(),
            'acc': best_acc,
            'acc_save':acc_save,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'args':args,
    }
    torch.save(state,save_path+'/tmp.pth')
    if epoch % args.save_interval == 0 :
        torch.save(state,save_path+'/epoch_'+str(epoch)+'.pth')

for epoch in range(args.cur_epoch,args.epochs):
    print(best_acc)
    if args.warm_up:
        if epoch<5:
            optimizer.param_groups[0]['lr'] = 0.01
        elif epoch ==5:
            optimizer.param_groups[0]['lr'] = 0.1
    if epoch in args.step_epoch:
            optimizer.param_groups[0]['lr'] *= args.lr_decay
    if epoch == args.stop_epoch:
        optimizer.param_groups[0]['lr'] = args.stop_lr
    train(epoch)
    test(epoch)
np.savetxt(save_path +'/result_'+str(best_acc)+'.txt',np.array(acc_save),fmt='%s')
torch.save(best_model_weights, save_path+'/best_acc_model_'+str(best_acc)+'.pth')
