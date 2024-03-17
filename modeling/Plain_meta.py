import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.networks.backbone import build_feature_extractor, NET_OUT_DIM
import random
from torch.distributions import Beta

class Plain_Net(nn.Module):
    def __init__(self, cfg, backbone="resnet18"):
        super(Plain_Net, self).__init__()
        self.cfg = cfg
        self.in_c = NET_OUT_DIM[backbone]
        self.dist = Beta(torch.FloatTensor([2]), torch.FloatTensor([2]))

        self.topk_rate = 0.1
        self.vars = nn.ParameterList()
        w = nn.Parameter(torch.ones([1,512,1,1]))
        torch.nn.init.kaiming_normal_(w)
        self.vars.append(w)
        self.vars.append(nn.Parameter(torch.zeros([1])))

    def functional_conv_block(self, x, weights, biases):
        x = F.conv2d(x, weights, biases, stride=1, padding=0)
        return x


    def forward(self, image=None, image_scale=None, var = None):
        if var is None:
            var = self.vars
        var = list(var)

        image_pyramid = list()
        for s in range(self.cfg.n_scales):
            if s == 0:
                feature = image
            else:
                feature = image_scale

            w, b = var[0], var[1]
            x = F.conv2d(feature, w, b, stride=1, padding=0)
            x = x.view(int(x.size(0)), -1)
            topk = max(int(x.size(1) * self.topk_rate), 1)
            x = torch.topk(torch.abs(x), topk, dim=1)[0]
            scores = torch.mean(x, dim=1).view(-1, 1)

            image_pyramid.append(scores)
        scores = torch.cat(image_pyramid, dim=1)  # 按维度1拼接
        score = torch.mean(scores, dim=1)

        return score.view(-1, 1)

    def mixup_data(self, xs, ys, xq):
        query_size = xq.shape[0]

        shuffled_index = torch.randperm(query_size)

        xs = xs[shuffled_index]
        ys = ys[shuffled_index]
        lam = self.dist.sample().cuda()
        mixed_x = lam * xq + (1 - lam) * xs

        return mixed_x, ys, lam

    def forward_metamix(self, x, x_scale, y, x2, x2_scale, y2, var):
        if var is None:
            var = self.vars
        var = list(var)

        image_pyramid = list()
        for s in range(self.cfg.n_scales):
            if s == 0:
                x_mix, reweighted_y, lam = self.mixup_data(x, y, x2)
            else:
                x_mix, reweighted_y, lam = self.mixup_data(x_scale, y, x2_scale)

            x = self.functional_conv_block(x_mix, var[0], var[1])

            x = x.view(int(x.size(0)), -1)
            topk = max(int(x.size(1) * self.topk_rate), 1)
            x = torch.topk(torch.abs(x), topk, dim=1)[0]
            scores = torch.mean(x, dim=1).view(-1, 1)

            image_pyramid.append(scores)

        scores = torch.cat(image_pyramid, dim=1)
        score = torch.mean(scores, dim=1)

        return score.view(-1, 1), reweighted_y, lam

    def zero_grad(self, vars=None):
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars



