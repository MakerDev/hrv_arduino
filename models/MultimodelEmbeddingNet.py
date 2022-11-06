import os 
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from models.ResNet3D import ResNet
from models.Conv1DNet import Conv1dNetwork
import torch
import torch.nn as nn


class MultimodalEmbeddingNet(nn.Module):
    def __init__(self, resnet3d, conv1dnet, fc_size=1024, out_channel=7):
        super().__init__()

        self.resnet:ResNet = resnet3d
        self.conv1dnet:Conv1dNetwork = conv1dnet
        self.dropout = nn.Dropout(0.5)
        
        resnet_flat_size = self.resnet.flat_size
        conv_flat_size = self.conv1dnet.flat_size
        self.flat_size = resnet_flat_size + conv_flat_size

        self.classifier = nn.Sequential(
            nn.Linear(self.flat_size, fc_size),
            nn.ReLU(),
            nn.BatchNorm1d(fc_size),
            self.dropout,
            nn.Linear(fc_size, fc_size//2),
            nn.ReLU(),
            nn.BatchNorm1d(fc_size//2),
            self.dropout,
            nn.Linear(fc_size//2, out_channel),
            nn.ReLU()
        )

    def forward(self, x_resnet, x_conv1d):
        conv_net_feature = self.conv1dnet.get_feature(x_conv1d)
        batch_size = conv_net_feature.size(0)
        resnet_feature = self.resnet.get_feature(x_resnet).reshape(batch_size, -1)
        feature_embedding = torch.cat([resnet_feature, conv_net_feature], dim=1)
        
        x = self.classifier(feature_embedding)

        return x

