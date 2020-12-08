#投票
import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchvision.models as models

import sys
import math

class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        return out


class DenseNet(nn.Module):
    def __init__(self, nClasses,growthRate=12, depth=100, reduction=0.5,  bottleneck=True):
        super(DenseNet, self).__init__()
        self.reduction = reduction
        self.num_classes = nClasses
        nDenseBlocks = (depth-4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = 2*growthRate
        nstages  = []
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1,
                               bias=False)
        nstages.append(nChannels)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = Transition(nChannels, nOutChannels)
        nChannels = nOutChannels
        nstages.append(nChannels)
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Transition(nChannels, nOutChannels)
        nChannels = nOutChannels
        nstages.append(nChannels)
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nstages.append(nChannels)

        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans3 = Transition(nChannels, nOutChannels)

        # out_channels = int(sum(nstages[1:])*self.reduction)

        self.fc = nn.Linear(nOutChannels, nClasses)
        self.bn1 = nn.BatchNorm2d(nOutChannels)
        self.branch = nn.ModuleList([self._make_branch(p[0], p[1]) for p in [[[nstages[1],nstages[2]], 16],[[nstages[2],nstages[3]], 8]]])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)
    def _make_branch(self,inplanes,pool_size):
        branch = []
        if pool_size:
            branch.append(nn.Sequential(
                nn.BatchNorm2d(inplanes[1]),
                nn.ReLU(),
                nn.Conv2d(inplanes[1], int(inplanes[0]*self.reduction), kernel_size=1, bias=False),
            ))#conv1_1
            branch.append(nn.Sequential(
                nn.BatchNorm2d(inplanes[0]),
                nn.ReLU(),
                nn.Conv2d(inplanes[0], int(inplanes[0]*self.reduction), kernel_size=1, bias=False),
            ))  # conv1_2
            branch.append(nn.Sequential(
                nn.BatchNorm2d(inplanes[1]),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode = 'bilinear',align_corners=True),
            ))#upsample
            branch.append(nn.Sequential(
                nn.BatchNorm2d(int(inplanes[0]*self.reduction)),
                nn.ReLU(),
                nn.Conv2d(int(inplanes[0]*self.reduction), int(inplanes[0]*self.reduction), kernel_size=3, stride=int(pool_size/4), padding=1, bias=False),
                nn.AdaptiveAvgPool2d(1),
            ))#conv3
            branch.append(nn.Linear(int(inplanes[0]*self.reduction),self.num_classes))
        return nn.ModuleList(branch)

    def forward(self, x):
        out = []
        branch_x = []
        x = self.conv1(x)
        x = self.trans1(self.dense1(x))
        branch_x.append(x)
        x = F.avg_pool2d(x, 2)
        x = self.trans2(self.dense2(x))
        branch_x.append(x)
        x = F.avg_pool2d(x, 2)
        x = self.dense3(x)
        branch_x.append(x)
        for i, x_ in enumerate(branch_x[:-1]):
            x_ = self.branch[i][1](x_) + self.branch[i][0](self.branch[i][2](branch_x[i + 1]))
            # x_ = torch.cat([self.branch[i][1](x_),self.branch[i][2](self.branch[i][0](branch_x[i + 1]))],1)
            x_ = self.branch[i][3](x_)
            x_ = x_.view(x_.size(0), -1)
            x_ = self.branch[i][4](x_)
            out.append(x_)

        x = self.trans3(x)
        x = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(x)), 8))
        x = self.fc(x)
        out.append(x)
        return out
def densenet100bc(num_classes):
    return DenseNet(growthRate=12, depth=100, reduction=0.5, nClasses=num_classes, bottleneck=True)
def densenet250bc(num_classes):
    return DenseNet(growthRate=24, depth=250, reduction=0.5, nClasses=num_classes, bottleneck=True)
if __name__ == '__main__':
    x = torch.randn((1,3,32,32,))
    model = DenseNet(growthRate=24,depth=250,reduction=0.5,nClasses=100,bottleneck=True)
    print(len(model(x)))
    print('# generator parameters:', sum(param.numel() for param in model.parameters()) / (1024 * 1024))