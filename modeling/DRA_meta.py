import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.networks.backbone import build_feature_extractor, NET_OUT_DIM
import copy

class DRA(nn.Module):
    def __init__(self, cfg, backbone="resnet18", flag = None):
        super(DRA, self).__init__()
        self.cfg = cfg
        self.flag = flag
        self.in_c = NET_OUT_DIM[backbone]
        self.vars = nn.ParameterList()
        self.topk_rate = 0.1
        self.drop = nn.Dropout(0)

        w = nn.Parameter(torch.ones([256, 512]))
        torch.nn.init.kaiming_normal_(w)
        self.vars.append(w)
        self.vars.append(nn.Parameter(torch.zeros([256])))

        w = nn.Parameter(torch.ones([1, 256]))
        torch.nn.init.kaiming_normal_(w)
        self.vars.append(w)
        self.vars.append(nn.Parameter(torch.zeros([1])))

        w = nn.Parameter(torch.ones([1, 512, 1, 1]))
        torch.nn.init.kaiming_normal_(w)
        self.vars.append(w)
        self.vars.append(nn.Parameter(torch.zeros([1])))

        w = nn.Parameter(torch.ones([1, 512, 1, 1]))
        torch.nn.init.kaiming_normal_(w)
        self.vars.append(w)
        self.vars.append(nn.Parameter(torch.zeros([1])))

        w = nn.Parameter(torch.ones([1, 512, 1, 1]))
        torch.nn.init.kaiming_normal_(w)
        self.vars.append(w)
        self.vars.append(nn.Parameter(torch.zeros([1])))

        w = nn.Parameter(torch.ones([512, 512, 3, 3]))
        torch.nn.init.kaiming_normal_(w)
        self.vars.append(w)
        self.vars.append(nn.Parameter(torch.zeros([512])))
        self.vars.append(nn.Parameter(torch.zeros([512])))
        self.vars.append(nn.Parameter(torch.zeros([512])))

    def forward(self, image=None, image_scale=None, label= None, var = None, st =False):
        if var is None:
            var = self.vars
        else:
            var =list(var)

        image_pyramid = list()
        for i in range(self.cfg.total_heads):
            image_pyramid.append(list())
        for s in range(self.cfg.n_scales):
            if s == 0:
                feature = image
            else:
                feature = image_scale

            ref_feature = feature[:self.cfg.nRef, :, :, :]  #self.cfg.nRef
            feature = feature[self.cfg.nRef:, :, :, :]

            if self.training and st == False:
                # normal_scores
                x = F.adaptive_avg_pool2d(feature, (1, 1))
                x = x.view(x.size(0), -1)
                w, b = var[0], var[1]
                x = self.drop(F.relu(F.linear(x, w, b)))
                w, b = var[2], var[3]
                normal_scores = F.linear(x, w, b)

                # abnormal_scores
                w, b = var[4], var[5]
                x = F.conv2d(feature[label != 2], w, b, stride=1, padding=0)
                x = x.view(int(x.size(0)), -1)
                topk = max(int(x.size(1) * self.topk_rate), 1)
                x = torch.topk(torch.abs(x), topk, dim=1)[0]
                abnormal_scores = torch.mean(x, dim=1).view(-1, 1)

                # dummy_scores
                w, b = var[6], var[7]
                x = F.conv2d(feature[label != 1], w, b, stride=1, padding=0)
                x = x.view(int(x.size(0)), -1)
                topk = max(int(x.size(1) * self.topk_rate), 1)
                x = torch.topk(torch.abs(x), topk, dim=1)[0]
                dummy_scores = torch.mean(x, dim=1).view(-1, 1)

                # comparison_scores
                ref = torch.mean(ref_feature, dim=0).repeat([feature.size(0), 1, 1, 1])
                temp = ref - feature
                x = temp
                w, b = var[10], var[11]
                x = F.conv2d(x, w, b, stride=3, padding=0)
                w, b = var[12], var[13]
                running_mean = nn.Parameter(torch.zeros([512]), requires_grad=False).cuda()
                running_var = nn.Parameter(torch.ones([512]), requires_grad=False).cuda()
                x = F.batch_norm(x,running_mean, running_var, weight=w, bias=b)
                x = F.relu(x)
                w, b = var[8], var[9]
                x = F.conv2d(x, w, b, stride=1, padding=0)
                x = x.view(int(x.size(0)), -1)
                topk = max(int(x.size(1) * self.topk_rate), 1)
                x = torch.topk(torch.abs(x), topk, dim=1)[0]
                comparison_scores = torch.mean(x, dim=1).view(-1, 1)

            else:
                # normal_scores
                x = F.adaptive_avg_pool2d(feature, (1, 1))
                x = x.view(x.size(0), -1)
                w, b = var[0], var[1]
                x = self.drop(F.relu(F.linear(x, w, b)))
                w, b = var[2], var[3]
                normal_scores = F.linear(x, w, b)

                # abnormal_scores
                w, b = var[4], var[5]
                x = F.conv2d(feature, w, b, stride=1, padding=0)
                x = x.view(int(x.size(0)), -1)
                topk = max(int(x.size(1) * self.topk_rate), 1)
                x = torch.topk(torch.abs(x), topk, dim=1)[0]
                abnormal_scores = torch.mean(x, dim=1).view(-1, 1)

                # dummy_scores
                w, b = var[6], var[7]
                x = F.conv2d(feature, w, b, stride=1, padding=0)
                x = x.view(int(x.size(0)), -1)
                topk = max(int(x.size(1) * self.topk_rate), 1)
                x = torch.topk(torch.abs(x), topk, dim=1)[0]
                dummy_scores = torch.mean(x, dim=1).view(-1, 1)

                # comparison_scores
                ref = torch.mean(ref_feature, dim=0).repeat([feature.size(0), 1, 1, 1])
                temp = ref - feature
                x = temp
                w, b = var[10], var[11]
                x = F.conv2d(x, w, b, stride=3, padding=0)
                w, b = var[12], var[13]
                running_mean = nn.Parameter(torch.zeros([512]), requires_grad=False).cuda()
                running_var = nn.Parameter(torch.ones([512]), requires_grad=False).cuda()
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b)
                x = F.relu(x)
                w, b = var[8], var[9]
                x = F.conv2d(x, w, b, stride=1, padding=0)
                x = x.view(int(x.size(0)), -1)
                topk = max(int(x.size(1) * self.topk_rate), 1)
                x = torch.topk(torch.abs(x), topk, dim=1)[0]
                comparison_scores = torch.mean(x, dim=1).view(-1, 1)

            for i, scores in enumerate([normal_scores, abnormal_scores, dummy_scores, comparison_scores]):
                image_pyramid[i].append(scores)
        for i in range(self.cfg.total_heads):
            image_pyramid[i] = torch.cat(image_pyramid[i], dim=1)
            image_pyramid[i] = torch.mean(image_pyramid[i], dim=1)
        return image_pyramid


