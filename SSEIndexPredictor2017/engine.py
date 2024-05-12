import json
import logging
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
import torch.optim as optim

import optuna
from functools import partial

from model import LSTMModel, GRUModel
from data_preprocessing import create_dataloader


best_score = float("inf")


def train_model(model, train_loader, num_epochs, lr):
    """训练模型"""

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

        logging.info(f"Epoch {epoch+1}/{num_epochs} loss: {loss.item()}")


def evaluate_model(model, test_loader, scaler):
    """模型预测结果"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    predictions = []
    values = []
    with torch.no_grad():
        for seq, labels in test_loader:
            seq, labels = seq.to(device), labels.to(device)

            y_pred = model(seq)

            y_pred = y_pred.cpu().numpy()  # 如果在GPU上运行，先移到CPU
            labels = labels.cpu().numpy()

            # 将预测和标签重塑为二维数组以进行反归一化
            y_pred = y_pred.reshape(-1, 1)  # 假设每个时间步只有一个特征
            labels = labels.reshape(-1, 1)

            y_pred = scaler.inverse_transform(y_pred)
            labels = scaler.inverse_transform(labels)

            predictions.append(y_pred.reshape(-1))
            values.append(labels.reshape(-1))

    # print(predictions, values, sep="\n")
    return predictions, values


def evaluate_reports(predictions, true_values):
    """计算预测结果各个指标"""
    # 初始化指标列表
    mses, rmses, maes, direction_accuracies = [], [], [], []

    # 对每个子列表进行计算
    for preds, trues in zip(predictions, true_values):
        mse, rmse, mae, direction_accuracy = evaluate_one_reports(
            preds, trues
        )

        # 将计算结果添加到指标列表中
        mses.append(mse)
        rmses.append(rmse)
        maes.append(mae)
        direction_accuracies.append(direction_accuracy)

    return mses, rmses, maes, direction_accuracies


def evaluate_one_reports(preds, trues):
    """计算预测结果各个指标"""
    mse = mean_squared_error(trues, preds)  # 计算均方误差
    rmse = np.sqrt(mse)  # 计算均方根误差
    mae = mean_absolute_error(trues, preds)  # 计算平均绝对误差
    direction_accuracy = calculate_direction_accuracy(trues, preds)  # 计算方向准确性

    return mse, rmse, mae, direction_accuracy


def calculate_direction_accuracy(true_values, predictions):
    """计算方向准确性"""
    correct_directions = 0
    for i in range(1, len(true_values)):
        true_direction = np.sign(true_values[i] - true_values[i - 1])
        pred_direction = np.sign(predictions[i] - predictions[i - 1])
        if true_direction == pred_direction:
            correct_directions += 1
    direction_accuracy = correct_directions / (len(true_values) - 1)
    return direction_accuracy


def train_and_evaluation(
    scaled_train_data,
    scaled_val_data,
    tw,
    batch_size,
    model,
    hidden_dim,
    num_layers,
    lr,
    num_epochs,
    scaler,
):
    """在训练集上训练，验证集上调整超参数"""

    # 创建DataLoader
    train_loader = create_dataloader(scaled_train_data, tw=tw, batch_size=batch_size)
    val_loader = create_dataloader(scaled_val_data, tw=tw, batch_size=1)

    if model == "LSTM":
        # 初始化模型
        model = LSTMModel(
            input_dim=1,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
    elif model == "GRU":
        # 初始化模型
        model = GRUModel(
            input_dim=1,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

    # 训练模型
    train_model(model, train_loader, num_epochs=num_epochs, lr=lr)

    # 评估模型
    predictions, values = evaluate_model(model, val_loader, scaler)

    mses = evaluate_reports(predictions, values)[0]
    score = sum(mses)

    return score, model


def objective(trial, scaled_train_data, scaled_val_data, scaler, best_model_path, num_epochs):
    """优化目标"""
    global best_score

    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    model_choice = trial.suggest_categorical("model", ["LSTM", "GRU"])
    tw = trial.suggest_categorical("tw", [40, 50, 60])

    # 运行模型训练和评估函数
    score, trained_model = train_and_evaluation(
        scaled_train_data=scaled_train_data,
        scaled_val_data=scaled_val_data,
        tw=tw,
        batch_size=batch_size,
        model=model_choice,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        lr=lr,
        num_epochs=num_epochs,
        scaler=scaler,
    )

    # 如果当前模型的分数比迄今为止的最佳分数好，更新最佳分数并保存模型
    if score < best_score:
        best_score = score
        torch.save(trained_model.state_dict(), best_model_path)
        logging.info(f"New best model saved with score: {score}")

    return score


def tune(
    scaled_train_data,
    scaled_val_data,
    scaler,
    best_model_path,
    best_model_params_path,
    n_trials,
    num_epochs
):
    """超参数调整"""

    # 创建一个 Optuna优化器
    study = optuna.create_study(direction="minimize")
    objective_partial = partial(
        objective,  # 优化目标函数
        scaled_train_data=scaled_train_data,
        scaled_val_data=scaled_val_data,
        scaler=scaler,
        best_model_path=best_model_path,
        num_epochs=num_epochs
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
    model_params,
    model_path,
    scaled_test_data,
    scaler,
):
    """测试集上的评估结果"""

    # 根据 model_params 创建模型
    if model_params["model"] == "LSTM":
        model = LSTMModel(
            input_dim=1,
            hidden_dim=model_params["hidden_dim"],
            num_layers=model_params["num_layers"],
        )
    elif model_params["model"] == "GRU":
        model = GRUModel(
            input_dim=1,
            hidden_dim=model_params["hidden_dim"],
            num_layers=model_params["num_layers"],
        )

    # 加载保存的模型权重
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    # 将模型设置为评估模式
    model.eval()

    # 创建测试 DataLoader
    test_loader = create_dataloader(
        scaled_test_data, tw=model_params["tw"], batch_size=1
    )

    # 使用模型进行预测
    predictions, true_values = evaluate_model(model, test_loader, scaler)

    preds, values = predictions[-1], true_values[-1]

    logging.info("Preds       Values")
    for pred, val in zip(preds, values):
        logging.info(f"{pred:<17.3f} {val:.3f}")
    logging.info("\n")

    # 评估预测结果
    mse, rmse, mae, direction_accuracy = evaluate_one_reports(preds, values)

    # 打印评估报告
    logging.info(
        f"评估报告:\n MSE: {mse}\n RMSE: {rmse}\n MAE: {mae}\n 方向准确性: {direction_accuracy}"
    )
