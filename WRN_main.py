"""
    PyTorch training code for Wide Residual Networks:
    http://arxiv.org/abs/1605.07146

    The code reproduces *exactly* it's lua version:
    https://github.com/szagoruyko/wide-residual-networks

    2016 Sergey Zagoruyko
"""

import argparse
import os
import json
import numpy as np
from tqdm import tqdm
import torch
from torch.optim import SGD
import torch.utils.data
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchnet as tnt
from torchnet.engine import Engine
from lib.utils import cast, data_parallel, print_tensor_dict,soft_cross_entropy
from lib.utils import Collabrative_Loss as CL
from torch.backends import cudnn
from model.WRN_resnet import resnet


cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Wide Residual Networks')
# Model options
parser.add_argument('--model', default='resnet', type=str)
parser.add_argument('--depth', default=28, type=int)
parser.add_argument('--width', default=10, type=float)
parser.add_argument('--dataset', default='CIFAR100', type=str)
parser.add_argument('--dataroot', default='/home/zyy/data/CIFAR100', type=str)
parser.add_argument('--dtype', default='float', type=str)
parser.add_argument('--groups', default=1, type=int)
parser.add_argument('--nthread', default=4, type=int)
parser.add_argument('--seed', default=1, type=int)

# Training options
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--epochs', default=240, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--weight_decay', default=0.0005, type=float)
parser.add_argument('--epoch_step', default='[60,120,200]', type=str,
                    help='json list with epochs to drop lr on')
parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--note', default='', type=str)

# Device options
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--save', default='WRN', type=str,
                    help='save parameters and logs in this folder')
parser.add_argument('--ngpu', default=2, type=int,
                    help='number of GPUs to use for training')
parser.add_argument('--gpu_id', default=['1'], type=list,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

def create_dataset(opt, train):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                    np.array([63.0, 62.1, 66.7]) / 255.0),
    ])
    if train:
        transform = T.Compose([
            T.Pad(4),
            T.RandomHorizontalFlip(),
            T.RandomCrop(32),
            transform
        ])
    return getattr(datasets, opt.dataset)(opt.dataroot, train=train, download=False, transform=transform)

acc_save = []
best_acc = 0.0
total = 0.0
correct = [0.0, 0.0, 0.0, 0.0]
is_test = False
stop_epoch = 140
soft_loss = soft_cross_entropy(num_classes = 100)
def main():
    global best_acc, acc_save
    opt = parser.parse_args()
    print('parsed options:', vars(opt))
    epoch_step = json.loads(opt.epoch_step)
    num_classes = 10 if opt.dataset == 'CIFAR10' else 100

    torch.manual_seed(opt.seed)
    #os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id[0],opt.gpu_id[1],opt.gpu_id[2]
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

    def create_iterator(mode):
        return DataLoader(create_dataset(opt, mode), opt.batch_size, shuffle=mode,
                          num_workers=opt.nthread, pin_memory=torch.cuda.is_available())

    train_loader = create_iterator(True)
    test_loader = create_iterator(False)

    f, params = resnet(opt.depth, opt.width, num_classes)

    def create_optimizer(opt, lr):
        print('creating optimizer with lr = ', lr)
        return SGD([v for v in params.values() if v.requires_grad], lr, momentum=0.9, weight_decay=opt.weight_decay)

    optimizer = create_optimizer(opt, opt.lr)

    epoch = 0
    if opt.resume != '':
        state_dict = torch.load(opt.resume)
        epoch = state_dict['epoch']
        params_tensors = state_dict['params']
        for k, v in params.items():
            v.data.copy_(params_tensors[k])
        optimizer.load_state_dict(state_dict['optimizer'])

    print('\nParameters:')
    print_tensor_dict(params)

    n_parameters = sum(p.numel() for p in params.values() if p.requires_grad)
    print('\nTotal number of parameters:', n_parameters)

    meter_loss = tnt.meter.AverageValueMeter()
    classacc = tnt.meter.ClassErrorMeter(accuracy=True)
    timer_train = tnt.meter.TimeMeter('s')
    timer_test = tnt.meter.TimeMeter('s')

    if not os.path.exists(opt.save):
        os.mkdir(opt.save)

    def h(sample):
        global correct,acc_save,total,is_test,stop_epoch
        inputs = cast(sample[0], opt.dtype)
        targets = cast(sample[1], 'long')
        y = data_parallel(f, inputs, params, sample[2], list(range(opt.ngpu)))
        loss = []
        predicts = []

        for o in y:
            if epoch < stop_epoch:
                loss.append(F.cross_entropy(o, targets))
            predicts.append(o.max(1)[1])
        loss = torch.sum(torch.stack(loss,dim=0)).float()
        if epoch >= stop_epoch:
            pre_targets = y[-1]
            loss += soft_loss(y[0],targets,torch.nn.Softmax(dim=1)(pre_targets))
        for i,x in enumerate(y[:-1]):
            loss += sum([CL(y[i],y[j]) for j in range(i+1,len(y))])
        if is_test:
            total += targets.size(0)
            correct[len(correct) - 1] += torch.mode(torch.stack(predicts), 0)[0].eq(targets).sum().item()
            for i, pre in enumerate(predicts):
                correct[i] += pre.eq(targets).sum().item()
        return loss,y[-1]
    def log(t, state):
        torch.save(dict(params=params, epoch=t['epoch'], optimizer=state['optimizer'].state_dict()),
                   os.path.join(opt.save, 'model.pt7'))
        z = {**vars(opt), **t}
        with open(os.path.join(opt.save, 'log.txt'), 'a') as flog:
            flog.write('json_stats: ' + json.dumps(z) + '\n')
        print(z)

    def on_sample(state):
        state['sample'].append(state['train'])

    def on_forward(state):
        loss = float(state['loss'])
        classacc.add(state['output'].data, state['sample'][1])
        meter_loss.add(loss)
        if state['train']:
            state['iterator'].set_postfix(loss=loss)

    def on_start(state):
        state['epoch'] = epoch

    def on_start_epoch(state):
        global correct,total,stop_epoch
        classacc.reset()
        meter_loss.reset()
        timer_train.reset()
        state['iterator'] = tqdm(train_loader, dynamic_ncols=True)
        correct = [0.0, 0.0, 0.0, 0.0]
        total = 0.0
        epoch = state['epoch'] + 1
        if epoch in epoch_step:
            lr = state['optimizer'].param_groups[0]['lr']
            state['optimizer'] = create_optimizer(opt, lr * opt.lr_decay_ratio)
        if epoch == stop_epoch:
            lr = state['optimizer'].param_groups[0]['lr']
            state['optimizer'] = create_optimizer(opt, lr * 10)

    def on_end_epoch(state):
        global  acc_save,correct,total,is_test,best_acc
        train_loss = meter_loss.value()
        train_acc = classacc.value()
        train_time = timer_train.value()
        meter_loss.reset()
        classacc.reset()
        timer_test.reset()

        with torch.no_grad():
            is_test = True
            engine.test(h, test_loader)
            acc_save.append([correct[0] / total, correct[1] / total, correct[2] / total, correct[3] / total])
        is_test = False
        if best_acc< correct[2] / total:
            best_acc = correct[2] / total
        test_acc = classacc.value()[0]
        print(log({
            "train_loss": train_loss[0],
            "train_acc": train_acc[0],
            "test_loss": meter_loss.value()[0],
            "test_acc": test_acc,
            "epoch": state['epoch'],
            "num_classes": num_classes,
            "n_parameters": n_parameters,
            "train_time": train_time,
            "test_time": timer_test.value(),
        }, state))
        print('==> id: %s (%d/%d), test_acc: \33[91m%.2f\033[0m,my_test_acc:\33[91m%.2f\033[0m,best_acc:\33[91m%.2f\033[0m' %
              (opt.save, state['epoch'], opt.epochs, test_acc,correct[3] / total*100,best_acc*100))
    engine = Engine()
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.hooks['on_start'] = on_start
    engine.train(h, train_loader, opt.epochs, optimizer)


if __name__ == '__main__':
    main()
    np.savetxt("WRN-40-4.txt", np.array(acc_save), fmt='%s')
