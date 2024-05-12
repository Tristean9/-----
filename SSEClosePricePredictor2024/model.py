import torch.nn as nn
import torch.nn.functional as F


# 定义LSTM和GRU子模型
class LSTMSubmodel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2):
        super(LSTMSubmodel, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout
        )
        # 全连接层，映射到输出维度
        self.fc = nn.Linear(hidden_dim, 5)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out  # 取序列的最后一个时间步


class GRUSubmodel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2):
        super(GRUSubmodel, self).__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout
        )
        # 全连接层，映射到输出维度
        self.fc = nn.Linear(hidden_dim, 5)

    def forward(self, x):
        out, hn = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out  # 取序列的最后一个时间步


# 定义元学习器
class MetaLearner(nn.Module):
    def __init__(self, meta_hidden_dim1, meta_hidden_dim2):
        super(MetaLearner, self).__init__()
        self.fc1 = nn.Linear(10, meta_hidden_dim1)  
        self.fc2 = nn.Linear(meta_hidden_dim1, meta_hidden_dim2) 
        self.fc3 = nn.Linear(
            meta_hidden_dim2, 5
        ) 

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        final_out = self.fc3(x) 
        return final_out
