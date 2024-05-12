import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    # LSTM模型定义
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim  # 隐藏层维度
        self.num_layers = num_layers  # LSTM层的层数
        # LSTM层定义，batch_first=True表示输入输出的第一维为batch_size
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # 全连接层，映射到输出维度
        self.fc = nn.Linear(hidden_dim, 24)

    def forward(self, x):
        # 获取设备信息（CPU或GPU）
        device = x.device
        # 初始化隐藏状态和细胞状态，并移动到x的设备上
        h0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim, device=device
        ).requires_grad_()
        c0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim, device=device
        ).requires_grad_()
        # 前向传播LSTM
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # 取最后一时间步的输出进行全连接层计算
        out = self.fc(out[:, -1, :]).unsqueeze(-1)  # 在最后一个维度增加尺寸1

        return out  # 返回最终的输出


class GRUModel(nn.Module):
    # GRU模型定义
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim  # 隐藏层维度
        self.num_layers = num_layers  # GRU层的层数
        # GRU层定义，batch_first=True表示输入输出的第一维为batch_size
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        # 全连接层，映射到输出维度
        self.fc = nn.Linear(hidden_dim, 24)

    def forward(self, x):
        # 获取设备信息（CPU或GPU）
        device = x.device
        # 初始化隐藏状态，并移动到x的设备上
        h0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim, device=device
        ).requires_grad_()
        # 前向传播GRU
        out, hn = self.gru(x, h0.detach())
        # 取最后一时间步的输出进行全连接层计算
        out = self.fc(out[:, -1, :]).unsqueeze(-1)  # 在最后一个维度增加尺寸1

        return out  # 返回最终的输出
