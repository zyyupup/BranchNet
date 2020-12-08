import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch

class VGG(nn.Module):

    def __init__(self, features, num_classes=10,scale_factor = 1.0):
        super(VGG, self).__init__()
        self.num_classes = num_classes
        self.scale_factor = scale_factor
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )


        self.branch = nn.ModuleList([self._make_branch(p) for p in [64,128,256]])
        self._initialize_weights()
    def forward(self, x):
        branch_x = []
        out = []
        for idx, module in enumerate(self.features):
            x = module(x)
            if idx in [5,12,22,32]:
                branch_x.append(x)
        for i, x_ in enumerate(branch_x):
            if i < len(branch_x) - 1:
                x_ = self.branch[i][0](x_)+self.branch[i][1](branch_x[i + 1])
                x_ = self.branch[i][2](x_)
                x_ = x_.view(x_.size(0), -1)
                x_ = self.branch[i][3](x_)
                out.append(x_)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        out.append(x)
        return out
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
    def _make_branch(self,inplanes,pool_size):
        branch = []
        if pool_size:
            branch.append(nn.Sequential(
                nn.Conv2d(inplanes, 2*inplanes, kernel_size=1, bias=False),
                nn.BatchNorm2d(2*inplanes),
                nn.ReLU(),
            ))#conv1
            branch.append(nn.Sequential(
                nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True),
                nn.BatchNorm2d(2*inplanes),
                nn.ReLU()
            ))#upsample
            branch.append(nn.Sequential(
                nn.Conv2d(2*inplanes, 2*inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(2*inplanes),
                nn.ReLU(),
                nn.AvgPool2d(pool_size,stride=1),
            ))#conv3
            branch.append(nn.Linear(2*inplanes,self.num_classes))#fc
        return nn.ModuleList(branch)
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['A']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model


def vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model


def vgg13(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13']))
    return model


def vgg13_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']))
    return model


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model


def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model


def vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model


def vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model
if __name__ == '__main__':
    model = vgg16_bn()
    print("params:{}".format(sum(p.numel() for p in model.parameters())/(1024*1024)))
    N, channel, w, h = 5, 3, 32, 32
    y = torch.randint(0, 10, (32,))
    x_in = torch.randn(N, channel, w, h)
    out = model(x_in)
    print(len(out))