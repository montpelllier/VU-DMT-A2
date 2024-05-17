import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# 创建一个简单的多层感知器模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 第一个全连接层
        self.relu = nn.ReLU()  # ReLU激活函数
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, output_size)  # 第二个全连接层

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
