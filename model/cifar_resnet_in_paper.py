import torch.nn as nn
import math
import torch
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=100,scale_factor = 1.0):
        self.inplanes = 16
        self.scale_factor = scale_factor
        self.num_classes = num_classes
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.branch = nn.ModuleList([self._make_branch(inplanes) for inplanes in [16*block.expansion,32*block.expansion]])
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
            if block.__name__== 'BasicBlock':
                """
                For CIFAR10 ResNet paper uses option A.([20,32,44,56,110,1202])
                """
                downsample = LambdaLayer(lambda x:F.pad(x[:, :, ::2, ::2],(0, 0, 0, 0, planes // 4, planes // 4), "constant",0))
            elif block.__name__== 'Bottleneck':
                """
                For CIFAR10 ResNet paper uses option B.(164,1001)
                """
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
        if self.scale_factor == 1.0:
            branch.append(nn.Sequential())
        else:
            branch.append(nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
                nn.BatchNorm2d(outplanes),
                nn.ReLU(),
            ))
        branch.append(nn.Sequential(
            nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        ))#conv
        branch.append(nn.Linear(outplanes,self.num_classes))#fc
        return nn.ModuleList(branch)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        branch_x = []
        out = []
        x = self.layer1(x)  # 32*32
        branch_x.append(x)
        x = self.layer2(x)  # 16*16
        branch_x.append(x)
        x = self.layer3(x)  # 8*8
        branch_x.append(x)

        for i, x_ in enumerate(branch_x[:-1]):
            x_ = self.branch[i][2](x_)+self.branch[i][1](self.branch[i][0](branch_x[i + 1]))
            x_ = self.branch[i][3](x_)
            x_ = x_.view(x_.size(0), -1)
            x_ = self.branch[i][4](x_)
            out.append(x_)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        out.append(x)
        return out



def resnet20(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [3,3,3], **kwargs)
    return model
def resnet32(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [5,5,5], **kwargs)
    return model
def resnet110(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [18, 18, 18], **kwargs)
    return model
def resnet164(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [18, 18, 18], **kwargs)
    return model
def resnet1202(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [200, 200, 200], **kwargs)
    return model
def resnet1001(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [111, 111, 111], **kwargs)
    return model
if __name__ == '__main__':
    model = resnet164(num_classes = 10,size_of_branch = 0.5)
    #print(model)
    print('# generator parameters:', sum(param.numel() for param in model.parameters())/(1000*1000))
    N,channel,w,h = 32,3,32,32
    y = torch.randint(0,10,(32,))
    x = torch.randn(N,channel,w,h)
    out = model(x)
    print(len(out))
