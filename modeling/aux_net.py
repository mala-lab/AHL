import torch.nn as nn


class AUX_Model(nn.Module):
    def __init__(self, args):
        super(AUX_Model, self).__init__()
        self.input_dim = args.epochs
        self.input_dim2 = args.episode_num
        self.decoder = nn.Sequential(
            nn.Conv1d(self.input_dim, self.input_dim, kernel_size=1, stride=1),
            nn.BatchNorm1d(self.input_dim),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim2*self.input_dim, 1)
        )

    def forward(self, time_feature= None):
        x = self.decoder(time_feature)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output


