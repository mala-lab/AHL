import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
from einops import rearrange


class AUX_Model(nn.Module):
    def __init__(self):
        super(AUX_Model, self).__init__()

        self.lstm = nn.LSTM(6, 6, 2, batch_first=True, dropout=0.5, bidirectional=True)
        self.fc = nn.Linear(12, 6)
        self.relu = nn.ReLU()

    def forward(self, time_feature= None):
        h_t = (torch.rand(4, time_feature.size(0), 6).cuda(), torch.rand(4, time_feature.size(0), 6).cuda())
        out, h_t = self.lstm(time_feature, h_t)
        out = self.fc(self.relu(out[:, -1]))

        return out


