import torch
import torch.nn.functional as F
import sys
sys.path.append("..")
import lib.utils as utils


def resnet(depth, width, num_classes):
    assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
    n = (depth - 4) // 6
    widths = [int(v * width) for v in (16, 32, 64)]

    def gen_block_params(ni, no):
        return {
            'conv0': utils.conv_params(ni, no, 3),
            'conv1': utils.conv_params(no, no, 3),
            'bn0': utils.bnparams(ni),
            'bn1': utils.bnparams(no),
            'convdim': utils.conv_params(ni, no, 1) if ni != no else None,
        }

    def gen_group_params(ni, no, count):
        return {'block%d' % i: gen_block_params(ni if i == 0 else no, no)
                for i in range(count)}
    def gen_branch_params(ni,no):
        return{
            'conv0':utils.conv_params(ni,no,1),
            'conv1':utils.conv_params(no,no,3),
            'bn0':utils.bnparams(ni),
            'bn1':utils.bnparams(no),
            'bn':utils.bnparams(no),
            'fc':utils.linear_params(no,num_classes)
        }

    flat_params = utils.cast(utils.flatten({
        'conv0': utils.conv_params(3, 16, 3),
        'group0': gen_group_params(16, widths[0], n),
        'branch1':gen_branch_params(widths[1],widths[0]),#------------branch1--------------
        'group1': gen_group_params(widths[0], widths[1], n),
        'branch2':gen_branch_params(widths[2],widths[1]),#-------------branch2------------
        'group2': gen_group_params(widths[1], widths[2], n),
        'bn': utils.bnparams(widths[2]),
        'fc': utils.linear_params(widths[2], num_classes),
    }))

    utils.set_requires_grad_except_bn_(flat_params)

    def block(x, params, base, mode, stride):
        o1 = F.relu(utils.batch_norm(x, params, base + '.bn0', mode), inplace=True)
        y = F.conv2d(o1, params[base + '.conv0'], stride=stride, padding=1)
        o2 = F.relu(utils.batch_norm(y, params, base + '.bn1', mode), inplace=True)
        z = F.conv2d(o2, params[base + '.conv1'], stride=1, padding=1)
        if base + '.convdim' in params:
            return z + F.conv2d(o1, params[base + '.convdim'], stride=stride)
        else:
            return z + x

    def group(o, params, base, mode, stride):
        for i in range(n):
            o = block(o, params, '%s.block%d' % (base,i), mode, stride if i == 0 else 1)
        return o

    def f(input, params, mode):
        out = []
        branch_x = []
        x = F.conv2d(input, params['conv0'], padding=1)
        g0 = group(x, params, 'group0', mode, 1)#16*32*32
        branch_x.append(g0)
        g1 = group(g0, params, 'group1', mode, 2)#32*16*16
        branch_x.append(g1)
        g2 = group(g1, params, 'group2', mode, 2)#64*8*8
        branch_x.append(g2)
        for i,x_ in enumerate(branch_x[:-1]):
            o_1 = F.relu(utils.batch_norm(branch_x[i+1],params,"branch"+str(i+1)+".bn0",mode),inplace=True)
            o_1 = F.upsample(o_1,scale_factor=2,mode='bilinear',align_corners=True)
            x_ = x_ + F.conv2d(o_1,params["branch"+str(i+1)+".conv0"])#conv1(relu(bn(x+1)))下一尺寸输出降维升尺寸
            x_ = F.relu(utils.batch_norm(x_,params,"branch"+str(i+1)+".bn1",mode),inplace=True)
            x_ = F.conv2d(x_,params["branch"+str(i+1)+".conv1"],stride=1, padding=1)

            x_ = F.relu(utils.batch_norm(x_, params, "branch"+str(i+1)+".bn", mode))
            x_ = F.avg_pool2d(x_,int(32/(i+1)),1,0)
            x_ = x_.view(x_.size(0),-1)
            x_ = F.linear(x_,params["branch"+str(i+1)+".fc.weight"],params["branch"+str(i+1)+".fc.bias"])
            out.append(x_)


        o = F.relu(utils.batch_norm(g2, params, 'bn', mode))
        o = F.avg_pool2d(o, 8, 1, 0)
        o = o.view(o.size(0), -1)
        o = F.linear(o, params['fc.weight'], params['fc.bias'])
        out.append(o)
        return out

    return f, flat_params
if __name__ == '__main__':
    x = torch.randn((1,3,32,32)).cuda()
    f,params = resnet(28,10,10)
    f(x,params,True)