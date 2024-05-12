from data_preprocessing import load_data, normalize_data, split_data
from engine import tune, evaluate_on_test_set
import json
import os
import logging
import optuna

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
    optuna_logger = optuna.logging.get_logger('optuna')
    optuna_logger.setLevel(logging.INFO)
    optuna_logger.addHandler(ch)
    optuna_logger.addHandler(fh)

def main():

    # 配置日志
    configure_logging()

    # 现在可以在代码中使用 logging 来记录信息
    logging.info("Starting the process...")

    # 加载和预处理数据
    data = load_data("./data/SSE_data.csv")

    # 按时间分割训练集，验证集，测试集
    train_data, val_data, test_data = split_data(
        data,
        "2013-01-04",
        "2016-04-14",
        "2016-04-15",
        "2016-09-06",
        "2016-09-07",
        "2017-02-10",
    )

    # 在训练集上对数据进行归一化，以训练集归一化为统一尺度，在验证集和测试集上进行归一化
    scaled_train_data, scaler = normalize_data(train_data)
    scaled_val_data = scaler.transform(val_data.reshape(-1, 1))
    scaled_test_data = scaler.transform(test_data.reshape(-1, 1))

    # 最好的模型的保存路径
    best_model_path = "trained_models/best_model.pth"
    best_model_params_path = "trained_models/best_model_params.json"

    # 只要模型权重和模型超参数有一个不存在，就进行训练和超参数调整
    if not (os.path.exists(best_model_params_path) and os.path.exists(best_model_path)):
        logging.info("Training and hyperparameter tuning starts.")

        # 使用训练集和验证集进行模型训练和超参数调整，得到模型参数字典;
        tune(
            scaled_train_data=scaled_train_data,
            scaled_val_data=scaled_val_data,
            scaler=scaler,
            best_model_path=best_model_path,
            best_model_params_path=best_model_params_path,
            n_trials=100,  # 希望的超参数组合数
            num_epochs=100
        )

    # 从json文件读取数据，将json数据转换为字典
    with open(best_model_params_path, "r") as json_file:
        best_model_params = json.load(json_file)

    # 使用测试集在最好的模型上进行评估
    evaluate_on_test_set(
        model_params=best_model_params,
        model_path=best_model_path,
        scaled_test_data=scaled_test_data,
        scaler=scaler,
    )
    logging.info("Process finished.")


if __name__ == "__main__":
    main()
