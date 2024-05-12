import json
import os
import logging
import optuna
import pandas as pd

from data_preprocessing import (
    load_pre_process_data,
    normalize_data,
    split_data,
)

from engine import tune, evaluate_on_test_set

# 配置日志记录器
def configure_logging():
    # 创建 logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # 设置日志记录级别

    # 创建 formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 创建 console handler 并设置级别为 info
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)  # 添加 formatter 到 ch
    logger.addHandler(ch)  # 将 ch 添加到 logger

    # 创建 file handler，写入日志文件，设置级别为 info
    fh = logging.FileHandler("ModelTrainingLifecycle.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)  # 添加 formatter 到 fh
    logger.addHandler(fh)  # 将 fh 添加到 logger

    # 配置Optuna的logger
    optuna_logger = optuna.logging.get_logger("optuna")
    optuna_logger.setLevel(logging.INFO)
    optuna_logger.addHandler(ch)
    optuna_logger.addHandler(fh)


def main():

    # 配置日志
    configure_logging()

    # 现在可以在代码中使用 logging 来记录信息
    logging.info("Starting the process...")

    # 选择特定的列作为特征和标签
    feature_columns = [
        "开盘价(元)",
        "最高价(元)",
        "收盘价(元)",
        "最低价(元)",
        "成交量(万股)",
        "成交金额(万元)",
        "换手率(%)",
        "PE市盈率(TTM)",
    ]
    label_column = "日度对数收益率"

    # 加载和预处理数据
    data, final_date_close_price = load_pre_process_data(
        "./data/上证指数走势-历史数据-已更新至20240430.csv", feature_columns
    )

    # 按时间分割训练集，验证集，测试集
    train1_data, train2_data, val_data, test_data = split_data(
        data,
        "1990/12/19",
        "2017/8/10",
        "2017/8/11",
        "2020/12/16",
        "2020/12/17",
        "2024/3/1",
        "2024/3/2",
        "2024/5/10",
    )

    train1_X, train1_y = (
        train1_data[feature_columns].values,
        train1_data[label_column].values,
    )
    train2_X, train2_y = (
        train2_data[feature_columns].values,
        train2_data[label_column].values,
    )
    val_X, val_y = val_data[feature_columns].values, val_data[label_column].values
    test_X, test_y = test_data[feature_columns].values, val_data[label_column].values

    # 在训练集上对数据进行标准化，以训练集标准化为统一尺度，在验证集和测试集上进行标准化
    scaled_train1_X, scaler = normalize_data(train1_X)
    scaled_train2_X = scaler.transform(train2_X)
    scaled_val_X = scaler.transform(val_X)
    scaled_test_X = scaler.transform(test_X)

    # 最好的模型的保存路径
    best_model_dir = "trained_models"
    best_model_params_path = "trained_models/best_model_params.json"

    # 只要模型权重和模型超参数有一个不存在，就进行训练和超参数调整
    if not (os.path.exists(best_model_params_path) and os.path.exists(best_model_dir)):
        logging.info("Training and hyperparameter tuning starts.")

        # 使用训练集和验证集进行模型训练和超参数调整，得到模型参数字典;
        tune(
            scaled_train1_X=scaled_train1_X,
            train1_y=train1_y,
            scaled_train2_X=scaled_train2_X,
            train2_y=train2_y,
            scaled_val_X=scaled_val_X,
            val_y=val_y,
            best_model_dir=best_model_dir,
            best_model_params_path=best_model_params_path,
            n_trials=500, # 尝试500种超参数组合
            num_epochs=50, # 训练50轮次
        )

    # 从json文件读取模型参数数据，将json数据转换为字典
    with open(best_model_params_path, "r") as json_file:
        best_model_params = json.load(json_file)

    # 使用测试集在最好的模型上进行预测
    actual_close_prices = evaluate_on_test_set(
        model_params=best_model_params,
        model_dir=best_model_dir,
        scaled_test_X=scaled_test_X,
        test_y=test_y,
        final_date_close_price=final_date_close_price,
    )
    
    logging.info(f"actual_close_prices: {actual_close_prices}")
    
    # 写入预测结果
    df = pd.read_csv("./XXX-SSE-pre.csv")
    df['SSE_index'] = actual_close_prices
    df.to_csv("./XXX-SSE-pre.csv", index=False, float_format='%.2f')
    
    logging.info("Process finished.")


if __name__ == "__main__":
    main()
