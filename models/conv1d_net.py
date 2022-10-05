import torch.nn as nn
import torch

class Conv1dNetwork(nn.Module):
    def __init__(self, in_channel=1, seq_len=60000, out_channel=7, kernel_size=64, fc_size=1024, stride=16):
        super(Conv1dNetwork, self).__init__()
        self.conv1d_1 = nn.Conv1d(in_channels=in_channel,
                                  out_channels=16,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=1)

        self.conv1d_2 = nn.Conv1d(in_channels=16,
                                  out_channels=32,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=1)

        self.pool = nn.MaxPool1d(4)
        self.dropout = nn.Dropout(0.5)
        self.conv_net = nn.Sequential(
            self.conv1d_1,
            self.pool,
            self.conv1d_2,
            self.pool
        )

        flat_size = self.get_flat_size((1, seq_len), self.conv_net)

        self.fc_layer1 = nn.Linear(flat_size, fc_size)
        self.fc_layer2 = nn.Linear(fc_size, fc_size//2)
        self.fc_layer3 = nn.Linear(fc_size//2, out_channel)
        self.fc_net = nn.Sequential(
            self.fc_layer1,
            nn.ReLU(),
            self.dropout,
            self.fc_layer2,
            nn.ReLU(),
            self.dropout,
            self.fc_layer3,
            nn.ReLU()
        )

    def get_flat_size(self, input_shape, conv_net):
        f = conv_net(torch.Tensor((torch.ones(1, *input_shape))))
        return torch.flatten(f).shape[0]

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv_net(x)
        # x = x.transpose(1, 2)
        x = torch.flatten(x, 1)

        # x = self.dropout(x)
        x = self.fc_net(x)

        return x
