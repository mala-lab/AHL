import torch.nn as nn
from torchvision import models


class FeatureClsuer(nn.Module):
    def __init__(self):
        super(FeatureClsuer, self).__init__()
        self.net = models.resnet18(weights='ResNet18_Weights.DEFAULT')

    def forward(self, x):
        x = self.net.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.net.fc(x)
        return x

