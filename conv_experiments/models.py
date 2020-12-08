import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class CNNet(nn.Module):
    def __init__(self, n_series, n_output, pooling_strategy, dropout=0.2):
        super(Net, self).__init__()
        self.dropout = dropout
        
        self.conv1 = nn.Conv1d(in_channels=n_series, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3)
        
        if pooling_strategy == 'max':
            self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
            self.pool3 = nn.AdaptiveMaxPool1d(output_size=32)
        elif pooling_strategy == 'avg':
            self.pool1 = nn.AveragePool1d(kernel_size=2, stride=2)
            self.pool2 = nn.AveragePool1d(kernel_size=2, stride=2)
            self.pool3 = nn.AdaptiveAveragePool1d(output_size=32)
        
        self.fc = nn.Linear(32, n_output)
        
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

# partially from https://github.com/locuslab/TCN/blob/master/TCN/tcn.py

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class Block(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(Block, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [Block(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
