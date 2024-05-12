import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np



def load_data(file_path):
    '''导入数据'''
    
    data = pd.read_csv(file_path, index_col=0)
    data["tdays"] = pd.to_datetime(data["tdays"], format="%Y%m%d")
    return data


def normalize_data(data):
    '''数据归一化'''
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data.reshape(-1, 1))
    return data, scaler


def split_data(
    data,
    train_start_date,
    train_end_date,
    val_start_date,
    val_end_date,
    test_start_date,
    test_end_date,
):  
    '''根据日期，分割训练集，验证集和测试集'''
    train_data = data[
        (data["tdays"] >= train_start_date) & (data["tdays"] <= train_end_date)
    ]
    val_data = data[(data["tdays"] >= val_start_date) & (data["tdays"] <= val_end_date)]
    test_data = data[
        (data["tdays"] >= test_start_date) & (data["tdays"] <= test_end_date)
    ]
    return (
        train_data["SSE_index"].values,
        val_data["SSE_index"].values,
        test_data["SSE_index"].values,
    )


def create_dataloader(data, tw, batch_size):
    """从输入数据构建pytorch的dataloader"""

    inout_seq = []
    L = len(data)
    start_idx = max(0, L - tw - 24)  # 确保最后24个点被作为标签包含
    for i in range(start_idx, L - tw - 24 + 1):
        train_seq = data[i : i + tw]
        train_label = data[i + tw : i + tw + 24]
        inout_seq.append((train_seq, train_label))

    # 将输入和标签分别转换为张量
    seqs = torch.FloatTensor(np.array([s[0] for s in inout_seq]))
    labels = torch.FloatTensor(np.array([s[1] for s in inout_seq]))

    # 创建 TensorDataset
    dataset = TensorDataset(seqs, labels)

    # 使用 TensorDataset 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader
