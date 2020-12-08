import torch.nn as nn
import math
import torch
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
import numpy as np
from  torch.autograd import Variable


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000,scale_factor = 1.0):
        self.inplanes = 64
        self.scale_factor = scale_factor
        self.num_classes = num_classes
        self.block = block
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.conv1 = nn.Conv2d(3, 64, padding=1,kernel_size=3,bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        stages = [64,128,256,512]
        self.layer1 = self._make_layer(block, stages[0], layers[0])
        self.layer2 = self._make_layer(block, stages[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, stages[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, stages[3], layers[3], stride=2)
        self.branch = self.branch = nn.ModuleList([self._make_branch(inplanes) for inplanes in [stages[0]*block.expansion,stages[1]*block.expansion,stages[2]*block.expansion]])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512*block.expansion,num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    def _make_branch(self,inplanes):
        branch = []
        outplanes = int(self.scale_factor*inplanes)
        branch.append(nn.Sequential(
            nn.Upsample(scale_factor=2,mode = 'bilinear',align_corners=True),
            nn.BatchNorm2d(2*inplanes),                 
            nn.ReLU(),
         ))#upsample
        branch.append(nn.Sequential(
            nn.Conv2d(2*inplanes, outplanes , kernel_size=1, bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(),
        ))#conv1
        if self.scale_factor == 1:
            branch.append(nn.Sequential())
        else:
            branch.append(nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
                nn.BatchNorm2d(outplanes),
                nn.ReLU(),
            ))#conv2
        branch.append(nn.Sequential(
            nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        ))#conv3
        branch.append(nn.Linear(outplanes,self.num_classes))#fc
        return nn.ModuleList(branch)
    def branchOutput(self,branch_id,x_,x_1):
        i = branch_id
        x_ = self.branch[i][2](x_) + self.branch[i][1](self.branch[i][0](x_1))
        x_ = self.branch[i][3](x_)
        x_ = x_.view(x_.size(0), -1)
        x_ = self.branch[i][4](x_)
        return x_
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        branch_x=[]
        out = []
        x = self.layer1(x)
        pre_x = x
        x = self.layer2(x)
        out.append(self.branchOutput(0,pre_x,x))#branch unit 1
        pre_x = x
        x = self.layer3(x)
        out.append(self.branchOutput(1,pre_x,x))#branch unit 2
        pre_x = x
        x = self.layer4(x)
        out.append(self.branchOutput(2,pre_x,x))#branch unit 3
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        out.append(x)
        return out


def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2],**kwargs)
    return model


def resnet34(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(pretrained=False , **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3],**kwargs)
    return model


def resnet152(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model
if __name__ == '__main__':
    model = resnet101(num_classes=1000,size_of_branch=0.5)
    print(sum(p.numel() for p in model.parameters())/1024/1024)
    N,channel,w,h = 5,3,224,224
    x_in = torch.randn(N,channel,w,h)
    out = model(x_in)



