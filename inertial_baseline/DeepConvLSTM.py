# ------------------------------------------------------------------------
# DeepConvLSTM model based on architecture suggested by Ordonez and Roggen 
# https://www.mdpi.com/1424-8220/16/1/115
# ------------------------------------------------------------------------
# Adaption by: Anonymized
# E-Mail: anonymized
# ------------------------------------------------------------------------

from torch import nn
import torch


class DeepConvLSTM(nn.Module):
    def __init__(self, channels, classes, window_size, conv_kernels=64, conv_kernel_size=5, lstm_units=128, lstm_layers=2, dropout=0.5, feature_extract=None):
        super(DeepConvLSTM, self).__init__()

        self.conv1 = nn.Conv2d(1, conv_kernels, (conv_kernel_size, 1))
        self.conv2 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1))
        self.conv3 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1))
        self.conv4 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1))
        self.lstm = nn.LSTM(channels * conv_kernels, lstm_units, num_layers=lstm_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_units, classes)
        self.activation = nn.ReLU()
        self.final_seq_len = window_size - (conv_kernel_size - 1) * 4
        self.lstm_units = lstm_units
        self.classes = classes
        self.feature_extract = feature_extract

    def forward(self, x):
        x = x.unsqueeze(1)
        feat = None
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        if self.feature_extract == 'conv':
            feat = x[:, :, -1, :]
            feat = feat.reshape(feat.shape[0], -1)
        x = x.permute(2, 0, 3, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x, h = self.lstm(x)
        x = x[-1, :, :]
        x = x.view(-1, self.lstm_units)
        if self.feature_extract == 'lstm':
            feat = x
        x = self.dropout(x)
        if feat is not None:
            return self.classifier(x), feat
        else:      
            return self.classifier(x)
        


class ShallowDeepConvLSTM(nn.Module):
    def __init__(self, channels, classes, window_size, conv_kernels=64, conv_kernel_size=5, lstm_units=128, lstm_layers=2, dropout=0.5, feature_extract=None):
        super(ShallowDeepConvLSTM, self).__init__()

        self.conv1 = nn.Conv2d(1, conv_kernels, (conv_kernel_size, 1))
        self.conv2 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1))
        self.conv3 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1))
        self.conv4 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1))
        self.lstm = nn.LSTM(channels * conv_kernels, lstm_units, num_layers=lstm_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_units, classes)
        self.activation = nn.ReLU()
        self.final_seq_len = window_size - (conv_kernel_size - 1) * 4
        self.lstm_units = lstm_units
        self.classes = classes
        self.feature_extract = feature_extract

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x, h = self.lstm(x)
        x = x.view(-1, self.lstm_units)
        x = self.dropout(x)   
        x = self.classifier(x)
        out = x.view(-1, self.final_seq_len, self.classes)
        return out[:, -1, :]
