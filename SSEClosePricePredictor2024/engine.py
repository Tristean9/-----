import json
import os
import logging
import numpy as np
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim

import optuna
from functools import partial

from model import LSTMSubmodel, GRUSubmodel, MetaLearner
from data_preprocessing import create_sub_dataloader, create_meta_dataloader


best_score = float("inf")


def train_model(model, train_loader, num_epochs, lr):
    """训练两个子模型"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()
    loss_function = nn.MSELoss()
    optimzer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.to(device)

            optimzer.zero_grad()
            y_pred = model(seq)

            loss = loss_function(y_pred, labels)
            loss.backward()
            optimzer.step()

        # logging.info(f"Epoch {epoch+1}/{num_epochs} loss: {loss.item()}")


def evaluate_model(model, test_loader):
    """模型预测结果"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    with torch.no_grad():
        for seq, labels in test_loader:
            seq, labels = seq.to(device), labels.to(device)
            y_pred = model(seq)

    return y_pred, labels


def calculate_mse(predictions, true_values):
    mse = mean_squared_error(true_values, predictions)  # 计算均方误差
    return mse


def train_and_evaluation(
    scaled_train1_X,
    train1_y,
    scaled_train2_X,
    train2_y,
    scaled_val_X,
    val_y,
    tw,
    batch_size,
    hidden_dim,
    num_layers,
    lr,
    num_epochs,
    meta_hidden_dim1,
    meta_hidden_dim2,
):
    """在训练集上训练，验证集上调整超参数"""

    # 创建DataLoader
    train1_loader = create_sub_dataloader(
        scaled_train1_X, train1_y, tw=tw, batch_size=batch_size
    )
    train2_loader = create_sub_dataloader(
        scaled_train2_X, train2_y, tw=tw, batch_size=batch_size
    )
    val_loader = create_sub_dataloader(scaled_val_X, val_y, tw=tw, batch_size=1)

    lstmSubmodel = LSTMSubmodel(
        input_dim=8, hidden_dim=hidden_dim, num_layers=num_layers, dropout=0.2
    )
    gruSubmodel = GRUSubmodel(
        input_dim=8, hidden_dim=hidden_dim, num_layers=num_layers, dropout=0.2
    )

    # 训练两个子模型
    train_model(
        model=lstmSubmodel,
        train_loader=train1_loader,
        num_epochs=num_epochs,
        lr=lr,
    )
    train_model(
        model=gruSubmodel,
        train_loader=train1_loader,
        num_epochs=num_epochs,
        lr=lr,
    )

    # 使用训练集2得到子模型的预测结果
    predictions1, labels = evaluate_model(model=lstmSubmodel, test_loader=train2_loader)
    predictions2, _ = evaluate_model(model=gruSubmodel, test_loader=train2_loader)

    # 使用子模型的预测结果创建元学习器的数据集
    meta_dataloader = create_meta_dataloader(
        predictions1=predictions1, predictions2=predictions2, labels=labels
    )

    metaLearner = MetaLearner(
        meta_hidden_dim1=meta_hidden_dim1, meta_hidden_dim2=meta_hidden_dim2
    )

    # 训练元学习器
    train_model(
        model=metaLearner,
        train_loader=meta_dataloader,
        num_epochs=num_epochs,
        lr=lr,
    )

    # 使用验证集得到评估结果
    final_predictions1, final_labels = evaluate_model(
        model=lstmSubmodel, test_loader=val_loader
    )
    final_predictions2, _ = evaluate_model(model=gruSubmodel, test_loader=val_loader)

    final_meta_dataloader = create_meta_dataloader(
        predictions1=final_predictions1,
        predictions2=final_predictions2,
        labels=final_labels,
    )

    final_predictions, final_labels = evaluate_model(
        model=metaLearner, test_loader=final_meta_dataloader
    )

    mse = calculate_mse(final_predictions.cpu().numpy(), final_labels.cpu().numpy())

    return mse, lstmSubmodel, gruSubmodel, metaLearner


def objective(
    trial,
    scaled_train1_X,
    train1_y,
    scaled_train2_X,
    train2_y,
    scaled_val_X,
    val_y,
    best_model_dir,
    num_epochs,
):
    """优化目标"""
    global best_score

    # 定义超参数网络
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    num_layers = trial.suggest_int("num_layers", 2, 4)
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
    meta_hidden_dim1 = trial.suggest_categorical("meta_hidden_dim1", [32, 64, 128, 256])
    meta_hidden_dim2 = trial.suggest_categorical("meta_hidden_dim2", [32, 64, 128, 256])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    tw = trial.suggest_categorical("tw", [20, 30, 40])

    # 运行模型训练和评估函数
    score, trained_sub_LSTMModel, trained_sub_GRUModel, trained_sub_MetaModel = (
        train_and_evaluation(
            scaled_train1_X=scaled_train1_X,
            train1_y=train1_y,
            scaled_train2_X=scaled_train2_X,
            train2_y=train2_y,
            scaled_val_X=scaled_val_X,
            val_y=val_y,
            num_epochs=num_epochs,
            tw=tw,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            lr=lr,
            meta_hidden_dim1=meta_hidden_dim1,
            meta_hidden_dim2=meta_hidden_dim2,
        )
    )

    # 如果当前模型的分数比迄今为止的最佳分数好，更新最佳分数并保存模型
    if score < best_score:
        best_score = score
        torch.save(
            trained_sub_LSTMModel.state_dict(),
            os.path.join(best_model_dir, "best_sub_LSTMModel"),
        )
        torch.save(
            trained_sub_GRUModel.state_dict(),
            os.path.join(best_model_dir, "best_sub_GRUModel"),
        )
        torch.save(
            trained_sub_MetaModel.state_dict(),
            os.path.join(best_model_dir, "best_MetaModel"),
        )
        logging.info(f"New best model saved with score: {score}")

    return score


def tune(
    scaled_train1_X,
    train1_y,
    scaled_train2_X,
    train2_y,
    scaled_val_X,
    val_y,
    best_model_dir,
    best_model_params_path,
    n_trials,
    num_epochs,
):
    """超参数调整"""

    # 创建一个 Optuna优化器
    study = optuna.create_study(direction="minimize")
    objective_partial = partial(
        objective,  # 优化目标函数
        scaled_train1_X=scaled_train1_X,
        train1_y=train1_y,
        scaled_train2_X=scaled_train2_X,
        train2_y=train2_y,
        scaled_val_X=scaled_val_X,
        val_y=val_y,
        best_model_dir=best_model_dir,
        num_epochs=num_epochs,
    )

    # 开始优化，这里的n是优化过程中尝试的参数组合数量
    study.optimize(objective_partial, n_trials=n_trials)

    # 打印最佳的参数
    logging.info(f"Number of finished trials: { len(study.trials)}")
    logging.info("Best trial:")
    trial = study.best_trial

    logging.info(f"Value: {trial.value}")
    logging.info("Params:")
    for key, value in trial.params.items():
        logging.info(f"    {key}: {value}")

    # 将字典保存到json文件
    with open(best_model_params_path, "w") as json_file:
        json.dump(trial.params, json_file, indent=4)

    return trial.params


def evaluate_on_test_set(
    model_params, model_dir, scaled_test_X, test_y, final_date_close_price
):
    """测试集上的评估结果"""

    # 根据 model_params 创建模型
    # 创建DataLoader

    test_loader = create_sub_dataloader(
        scaled_test_X, test_y, tw=model_params["tw"], batch_size=1
    )

    lstmSubmodel = LSTMSubmodel(
        input_dim=8,
        hidden_dim=model_params["hidden_dim"],
        num_layers=model_params["num_layers"],
        dropout=0.2,
    )
    gruSubmodel = GRUSubmodel(
        input_dim=8,
        hidden_dim=model_params["hidden_dim"],
        num_layers=model_params["num_layers"],
        dropout=0.2,
    )

    # 加载保存的模型权重
    sub_lstm_state_dict = torch.load(os.path.join(model_dir, "best_sub_LSTMModel"))
    lstmSubmodel.load_state_dict(sub_lstm_state_dict)

    sub_gru_state_dict = torch.load(os.path.join(model_dir, "best_sub_GRUModel"))
    gruSubmodel.load_state_dict(sub_gru_state_dict)

    predictions1, labels = evaluate_model(model=lstmSubmodel, test_loader=test_loader)
    predictions2, _ = evaluate_model(model=gruSubmodel, test_loader=test_loader)

    meta_dataloader = create_meta_dataloader(
        predictions1=predictions1, predictions2=predictions2, labels=labels
    )

    metaLearner = MetaLearner(
        meta_hidden_dim1=model_params["meta_hidden_dim1"],
        meta_hidden_dim2=model_params["meta_hidden_dim2"],
    )

    meta_leader_state_dict = torch.load(os.path.join(model_dir, "best_MetaModel"))
    metaLearner.load_state_dict(meta_leader_state_dict)

    final_predictions, _ = evaluate_model(
        model=metaLearner, test_loader=meta_dataloader
    )

    actual_close_prices = calculate_predicted_close_prices(
        final_predictions.flatten().cpu().tolist(), final_date_close_price
    )

    return actual_close_prices


def calculate_predicted_close_prices(predicted_log_returns, final_date_close_price):
    """
    计算预测的收盘价。

    参数：
    - predicted_log_returns：预测的日度对数收益率列表。
    - final_date_close_price：数据集中最后一天的收盘价。

    返回：
    - last_five_days_close_prices：最后五天的预测收盘价列表。
    """

    # 获取最后一个已知的对数收盘价
    final_date_log = np.log(final_date_close_price)

    # 初始化一个列表来存储每天的对数收盘价
    log_close_prices = [final_date_log]
    

    # 计算接下来五天的对数收盘价
    for log_return in predicted_log_returns:
        # 上一个收盘价的对数加上预测的日度对数收益率
        log_close_prices.append(log_close_prices[-1] + log_return)

    # 计算实际收盘价
    actual_close_prices = np.exp(log_close_prices[1:])  # 把对数收盘价转换回实际收盘价

    return actual_close_prices.reshape(-1)
