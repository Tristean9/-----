import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


def load_pre_process_data(file_path, feature_columns):
    """导入数据"""

    data = pd.read_csv(file_path)
    data["时间"] = pd.to_datetime(data["时间"], format="%Y/%m/%d")
    

    for column in feature_columns:
        # 先确保列是字符串类型
        data[column] = data[column].astype(str)
        data[column] = data[column].str.replace(",", "")  # 去除数值字符串中的逗号
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )  # 将列转换为数值型，非数值转为NaN
        mean_value = data[column].mean()  # 计算平均值
        data[column] = data[column].fillna(mean_value)  # 用平均值填充缺失值

    data["日度对数收益率"] = np.log(data["收盘价(元)"]).diff()
    
    final_date_close_price = data[data["时间"] == "2024/4/30"]["收盘价(元)"]

    # 创建一个包含预测日期的列表
    new_dates = ["2024/5/6", "2024/5/7", "2024/5/8", "2024/5/9", "2024/5/10"]

    # 创建一个新的 DataFrame，其中只有日期列被填充，其他列保留为空
    column_names = data.columns  # 假设你想保持和原始数据集相同的列结构
    new_data = pd.DataFrame(columns=column_names)

    # 填充日期列
    new_data["时间"] = pd.to_datetime(new_dates, format="%Y/%m/%d")

    # 将新的 DataFrame 附加到原始数据集下方
    data = pd.concat([data, new_data], ignore_index=True)

    return data, final_date_close_price


def normalize_data(data):
    """数据标准化"""

    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data, scaler


def split_data(
    data,
    train1_start_date,
    train1_end_date,
    train2_start_date,
    train2_end_date,
    val_start_date,
    val_end_date,
    test_start_date,
    test_end_date,
):
    """根据日期，分割训练集，验证集和测试集"""
    train1_data = data[
        (data["时间"] >= train1_start_date) & (data["时间"] <= train1_end_date)
    ]
    train2_data = data[
        (data["时间"] >= train2_start_date) & (data["时间"] <= train2_end_date)
    ]
    val_data = data[(data["时间"] >= val_start_date) & (data["时间"] <= val_end_date)]
    test_data = data[
        (data["时间"] >= test_start_date) & (data["时间"] <= test_end_date)
    ]
    return train1_data, train2_data, val_data, test_data


def create_sub_dataloader(input_X, input_y, tw, batch_size):
    """从输入数据构建pytorch的dataloader"""

    inout_seq = []
    L = len(input_X)
    start_idx = max(0, L - tw - 5)  # 确保最后5个点被作为标签包含
    for i in range(start_idx, L - tw - 5 + 1):
        train_seq = input_X[i : i + tw]
        train_label = input_y[i + tw : i + tw + 5]
        inout_seq.append((train_seq, train_label))

    # 将输入和标签分别转换为张量
    seqs = torch.FloatTensor(np.array([s[0] for s in inout_seq]))
    labels = torch.FloatTensor(np.array([s[1] for s in inout_seq]))

    # 创建 TensorDataset
    dataset = TensorDataset(seqs, labels)

    # 使用 TensorDataset 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader


def create_meta_dataloader(
    predictions1, predictions2, labels, batch_size=1, shuffle=True
):
    # 使用 torch.cat 在预测结果上合并张量
    features = torch.cat((predictions1, predictions2), dim=1)

    # 创建 TensorDataset
    dataset = TensorDataset(features, labels)

    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader
