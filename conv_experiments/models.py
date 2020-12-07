import torch
from torch import nn
import torch.nn.functional as F

class CNNet(nn.Module):
    def __init__(self, n_series, pooling_strategy, dropout=0.2):
        super(Net, self).__init__()
        self.dropout = dropout
        
        self.conv1 = nn.Conv1d(in_channels=n_series, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3)
        
        if pooling_strategy == 'max':
            self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
            self.pool3 = nn.AdaptiveMaxPool1d(output_size=32)
        elif pooling_strategy = 'avg':
            self.pool1 = nn.AveragePool1d(kernel_size=2, stride=2)
            self.pool2 = nn.AveragePool1d(kernel_size=2, stride=2)
            self.pool3 = nn.AdaptiveAveragePool1d(output_size=32)
        
        self.fc = nn.Linear(32, 1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.dropout(x, self.dropout)
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = F.dropout(x, self.dropout)
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.dropout(x, self.dropout)
        x = self.pool3(x)
        x = self.fc(x)
        return x

class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        # TODO
    def forward(self, x):
        # TODO
        pass

class TCNet(nn.Module):
    def __init__(self, n_series, dropout=0.2):
        super(TCNet, self).__init__()
        # TODO
        
    def forward(self, x)
        # TODO
        pass
